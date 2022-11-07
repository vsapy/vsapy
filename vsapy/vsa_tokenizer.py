import re
from vsapy.cspvec import *
from vsapy.helpers import *


class VsaTokenizer(object):
    def __init__(self, role_vecs, _usechunksforwords=True,
                 allow_skip_words=False, skip_words=None, skip_word_criterion=None,
                 use_word2vec=False, word2Vec_model=None, my_r2b=None):

        """
        This is a helper class in containing a number of parsers that can encode words, sentences and whole
        documents into a vsa vector or set of hierarchical vectors.

        :param symbol_dict:
        :param _usechunksforwords:
        :param allow_skip_words:
        :param skip_words:
        :param skip_word_criterion: function returning bool indicates whether to skip this word, can pass as a lambda
        :param use_word2vec:
        :param word2Vec_model: A word2vec semantic vector model.
        :param my_r2b:
        """
        self.role_vecs = role_vecs
        self.symbol_dict = role_vecs.symbol_dict
        self.use_word2vec = use_word2vec
        self.miss_from_word2vec = {}
        self.linecheck = []  # Used for debug to check vsa unbinding replays. Allows us to check if the unbinding
                             # reproduces the same document lines as were built duing encoding.
        self.total_word_count = 0  # Used to count the total number of words seen by the tokenizer
                                   # Reset this before starting a new document.

        if use_word2vec:
            assert word2Vec_model is not None, "you must supply a word2vec model"
            assert my_r2b is not None, "you must supply a RealToBinary converter"
            self.model = word2Vec_model
            self.my_r2b = my_r2b
        else:
            self.model = None  # word to vec model
            self.my_r2b = None  # real to binary converter

        self.allow_skips = allow_skip_words
        if allow_skip_words:
            assert skip_words is not None, "you must specify a skip words dictionary"
            self.skip_words = skip_words
        else:
            self.skip_words = {}

        self.meets_skip_words_criterion = skip_word_criterion

        self.seen_words = {}  # Tracks seen words so they do not need to be rebuilt

        self.usechunksforwords = _usechunksforwords
        if self.usechunksforwords:
            self.createWordVector = self.chunkWordVector  # Using our own chunking scheme, allows individual words as services
        else:
            self.createWordVector = self.gb_wordvector_as_chunk  # Graham's piBinding - faster

    @staticmethod
    def replace_non_alphanumerics(source, replacement_character='_'):
        result = re.sub("[^_' a-zA-Z0-9]", replacement_character, source)

        return result

    def chain_vecs(self, word):
        chain = np.roll(self.symbol_dict[word[0]], 1)
        for shift, c in enumerate(word[1:], 2):
            chain = vsa.bind(chain, np.roll(self.symbol_dict[c], shift))

        return chain

    def chunkWordVector(self, word):
        lettervecs = [self.symbol_dict[a] for a in word]

        cnk = CSPvec(word, lettervecs, self.role_vecs)
        if self.usechunksforwords:
            self.linecheck.append(word)  # record the word for verification

        return cnk

    def createWordVector_GB(self, word):
        shift = 0
        letter_vecs = []
        for c in word:
            shift += 1
            letter_vecs.append(np.roll(self.symbol_dict[c], shift))

        return vsapy.BagVec(letter_vecs).myvec

    def gb_wordvector_as_chunk(self, word):
        return CSPvec(word, [self.createWordVector_GB(word)], self.role_vecs)

    def get_word2Vec(self, w):
        '''
        Returns None if word not found in Model, otherwise BSC mapped from model's real number vector
        '''
        w1 = w
        v = None
        if w not in self.model:
            if w1[0] == "'":
                w1 = w[1:]  # Chop off a leading quote '
            if len(w1) > 0 and w1[-1] == "'":
                w1 = w[:-1]  # Chop off a trailing quote '
            if len(w1) > 2 and w[-2] == "'":
                w1 = w.replace("'d", "ed")
                w1 = w1.replace("'s", "")

        if w1 in self.model:
            v = self.my_r2b.to_bin(self.model[w1])

        return v

    def get_word_vector(self, w):
        v = None
        try:
            if self.use_word2vec:
                v = self.get_word2Vec(w)

            if v is None:
                v = self.createWordVector(w).myvec
                if self.use_word2vec:
                    # Count the number of words occurances not in the word2vec model
                    self.miss_from_word2vec[w] = self.miss_from_word2vec.get(w, 0) + 1

        except Exception as e:
            e = e

        return v

    def get_vector_or_skip_the_word(self, w):
        '''
        Returns BSC vector representation of the word or None if failed to build a vector
        (e.g word is in skip list or word contains all non-printing chars.
        '''
        v = None
        try:
            if self.allow_skips and (w in self.skip_words or self.meets_skip_words_criterion(w)):
                if w not in self.skip_words:
                    self.skip_words[w] = PackedVec(self.get_word_vector(w))
                return None

            if w in self.seen_words:
                if w in self.miss_from_word2vec:
                    self.miss_from_word2vec[w] += 1  # Count the occurrences of words missing from our word2vec model
                return self.seen_words[w].myvec
            else:
                v = self.get_word_vector(w)
                if v is not None:
                    self.seen_words[w] = PackedVec(v)  # We only build new word vectors once.

        except [ValueError, IndexError] as e:
            e = e

        return v

    def chunkSentenceVector(self, sentence1):

        db_sentence = None
        wordvecs = []
        try:
            sentence = sentence1.replace('--', ' ')
            sentence = VsaTokenizer.replace_non_alphanumerics(sentence, ' ')
            words = sentence.split()

            for w in words:
                w = w.strip()
                w = w.replace("'d", "ed")
                if w is None: continue

                vec = self.get_vector_or_skip_the_word(w)
                if vec is not None:
                    self.total_word_count += 1
                    wordvecs.append(CSPvec(w, [vec], self.role_vecs))

            if len(wordvecs) == 0:
                wordvecs = wordvecs  # For Debug
                return None
            else:
                try:
                    db_sentence = CSPvec.buildchunks(sentence, wordvecs, self.role_vecs,
                                                     split_tail_evenly=True, rebuild_names=True)
                except [ValueError, IndexError] as e:
                    print(e)
                    db_sentence = db_sentence  # For Debug

        except [ValueError, IndexError] as e:
            print(e)
            e = e  # For Debug

        return db_sentence

