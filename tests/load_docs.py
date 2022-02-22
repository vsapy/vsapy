from vsapy.logger_utils import *
from vsapy.bag import *
from vsapy.helpers import *


def load_vsa_document_set(usechunksforWords,
                          use_word2vec,
                          use_shaping,
                          no_acts=100, no_scenes_per_act=100,
                          prepend_level_labels=False):

    word_format = ''
    word_format += '_CW-T' if usechunksforWords else 'CW-F'
    word_format += '_WV-T' if use_word2vec else '_WV-F'
    word_format += '_PVECS' if usechunksforWords else '_CHAIN'
    word_format += '_SHAPED' if use_shaping else '_NoShape'

    load_fn = f'Models/hamlet_{word_format}.kv'
    save_fn = f'Models/hamlet_{word_format}_out.kv'

    file_path = './data/output_vec_files'

    runlist = [('macbeth', f'{file_path}/macbeth_nltk_{word_format}.bin', 'Macbeth', 'OE_tk_mbeth',
                'Macbeth: Old English. Natural Language Tool Kit.'),
               ('ntlk_shakespear', f'{file_path}/hamlet_nltk_{word_format}.bin', 'Hamlet', 'OE_tk_ham',
                'Hamlet: Old English. Natural Language Tool Kit.'),
               ('hamlet_orig', f'{file_path}/hamlet_stanzas_{word_format}.bin', 'Hamlet', 'OE_og_ham',
                'Hamlet: Old English. Original Source.'),
               ('nofear_old', f'{file_path}/hamlet_old_{word_format}.bin', 'Hamlet', 'OE_nf_ham',
                'Hamlet: Old English. NoFear Translation Source.'),
               ('nofear_new', f'{file_path}/hamlet_new_{word_format}.bin', 'Hamlet', 'NE_nf_ham',
                'Hamlet: New English. NoFear Translation Source.')]


    docs = {}
    for kk, fn, play_name, short_info, long_info in runlist:
        top_chunk = deserialise_object(fn, None)
        if prepend_level_labels:
            BareChunk.add_level_labels(top_chunk, 0, short_info)

        docs[short_info] = top_chunk


    return docs

if __name__ == '__main__':

    levels_to_extract = [0]
    docs = load_vsa_document_set(usechunksforWords=True, use_word2vec=False, use_shaping=False, prepend_level_labels=True)

    # compare documents at the document level
    print("Compare at the document level:")
    for k1, v1 in docs.items():
        for k2, v2 in docs.items():
            print(f"{k1}<->{k2} hsim={vsa.hsim(v1.myvec, v2.myvec):0.4f}")

    # compare documents at the scene level
    # This is an example of how we can did into the chunk heirarchy to compare things
    # It is a bit contrived because I am only comparing `like-with-like', i.e. scenes that I know should match.
    print("Compare at the document level:")
    doc_chunks = {}
    min_entries = 100000
    for k1, v1 in docs.items():
        chunks = []
        # pass a list of the levels you want to retrieve, in this case [0, 1] = the whole play and each act.
        # passing [2] would retrieve only the scene data.
        BareChunk.get_levels_as_list(v1, [0, 1], 0, chunks)
        doc_chunks[k1] = chunks
        min_entries = min(min_entries, len(chunks))

    for k1, v1 in docs.items():
        for k2, v2 in docs.items():
            if k1 == k2:
                continue
            print(f"\n\n{k1}<->{k2} ")
            for i1 in doc_chunks[k1]:
                i1_level = i1.get_level_number()
                for i2 in doc_chunks[k2]:
                    i2_level = i2.get_level_number()
                    if i1_level == i2_level:
                        print(f"hsim={vsa.hsim(i1.myvec, i2.myvec):0.4f} : {i1.aname}<->{i2.aname}")

    quit()
