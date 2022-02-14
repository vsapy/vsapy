"""
This demo build various versions of shakespeare plays, namly multiple versions fo hamlet and also an old-english
macbeth. The idea is to demo the recursive chunking.

The 'runlist' list (see below) details where to find each document source and what parser to use to convert the words to vectors.

Each parser processes words into vector, and the proceeds to perform heirarchical bundling of the word vecs into
     word_vecs->sentence_vecs->scene_vecs->act_vecs->whole_document_vec
         see CSPvec.buildchunks()

"""
import os
import random
from os.path import exists
import sys
import xmltodict as xml

from vsapy.logger_utils import *

import vsapy as vsa
from vsapy.vsatype import VsaType, VsaBase
from vsapy.bag import *

if sys.version_info[0] >= 3:
    iteritems = lambda x: iter(x.items())

sys.path.append('/home/chris/Documents/Graham-GenSim')

import nltk.data
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

from collections import defaultdict
import string
import json
import pandas as pd
import subprocess
import time
import operator
#from vectors import *
# from newt.wire.Adapter import Adapter
# from newt.wire.Transport import Transport
from threading import Timer
import logging
import time
import timeit
# import bitarray as ba
import pickle
import re
from operator import itemgetter
from collections import Counter
# from gensim.models.keyedvectors import KeyedVectors
# from gensim.models import Word2Vec, KeyedVectors

import numpy as np
# import Vector_Sum_Matching3 as kvec
from vsapy.cspvec import *
from vsapy.helpers import *

__author__ = 'simpkin'

#-----------------------------------------------------------------------------------------
# Should move these into the vsapy library and rename 'gb_wordvector_as_chunk()' and 'createWordVector_GB()'
#
def xorBind(A, B):
    '''
    A xor B is communtative.
    we cyclic shift A by 1 to make this binding non-commutative
    :param A: Vector1
    :param B: Vector2
    :return:  piA xor B
    '''
    V = np.logical_xor(np.roll(A, 1), B) * 1
    return V


def xorUnBind(A, B):
    V = np.roll(np.logical_xor(A, B), -1)
    return V

def gb_wordvector_as_chunk(word):
    return CSPvec(word, [createWordVector_GB(word)], -1, role_vecs)


def createWordVector_GB(word):
    v = symbol_dict[word[0]]
    for a in word[1:]:
        v = xorBind(v, symbol_dict[a])
    return v
#
#---------------------------------------------------------------------------------------


def fixup_chars(s):
    s = s.replace(u'\u2018', "'")
    s = s.replace(u'\u2019', "'")
    s = s.replace(u'\u2013', "-")  # u'\u2014' = '-'
    s = s.replace(u'\u2014', "-")  # u'\u2014' = '-'
    s = s.replace(u'\xe8', "e")  # replace e-umlat with 'e'
    s = s.replace(u'\u201c', '')  # u'\u201c' = '"'
    s = s.replace(u'\u201d', '')  # u'\u201d' = '"'
    s = s.replace(u'\xe0', 'a')
    s = s.replace(u'\u2003', ' ')
    s = s.replace(u'\u2026', ',')  # u'\u2026' = "..."
    s = s.replace('"', '')
    s = s.replace('\n', '')
    s = s.replace('\t', ' ')
    # Un-comment next to lines to debug unicode fails
    # print(s)
    # print(s.encode('ascii', 'replace'))
    return s.strip()


def get_act_number(act_string):
    act_string = act_string[4:]
    parts = act_string.split(".")
    return str(parts[0])


def get_scene_number(scene_string):
    act_string = scene_string[6:]
    parts = act_string.split(".")
    return str(parts[0])


def split_stanza_words(lns, splitters, maxwords):
    '''
    we want to split the stanza so the the number of words is less than maxWords
    we will split based on knowledge punctuation creating natural splits, then by maxwords

    :param lns: stanza to split
    :param splitters: an ordered list to try the split on, e.g. ['.', ';', ':', ',']
    :param maxwords: Max number of words in one section
    :return: array of lines
    '''

    newlines = []
    for l1 in lns:
        l1 = l1.replace("\n", "").strip()
        if len(l1.split()) > maxwords:
            if len(splitters) > 0:
                newlines.extend(
                    split_stanza_words([e + splitters[0] for e in l1.split(splitters[0]) if e], splitters[1:],
                                       maxwords))
                # newlines.extend(split_stanza_words(l1.split(splitters[0]), splitters[1:], maxwords))
            else:
                # The line is still too long but we have no more splitters
                # we will split the line by length
                tmp = l1.split()
                llen = len(tmp)
                m = 0
                while m + maxwords <= llen:
                    newlines.append(' '.join(tmp[m:m + maxwords]))
                    m += maxwords
                newlines.append(' '.join(tmp[m:llen]))  # grab any leftover chunk
        else:
            newlines.append(l1)

    return newlines


def get_longest_split_index(ln, splitchar, maxlen):
    newlines = []
    ii = 0
    for ii in reversed(range(len(maxlen))):
        if ln[ii] == splitchar: break

    if ii > 0:
        newlines.append(ln[:ii])
        if len(ln[ii + 1:]) <= maxlen:
            line_remainder = ln[ii + 1:]
    else:
        line_front = ''
        line_remainder = ln
    return ii, line_front, line_remainder


def get_key(key):
    try:
        return int(key)
    except ValueError:
        return key


def replace_non_alphanumerics(source, replacement_character='_'):
    result = re.sub("[^_' a-zA-Z0-9]", replacement_character, source)

    return result


def chunkWordVector(word):
    lettervecs = [symbol_dict[a] for a in word]

    cnk = CSPvec(word, lettervecs, -1, role_vecs)
    if usechunksforWords:
        linecheck.append(word)  # record the word for verification

    return cnk


class word_tracker(object):
    skip_words = {}
    seen_words = {}

    def __init__(self, use_word2vec, incl_all_words=True):
        pass

    @staticmethod
    def get_word2vec_vector(w):
        if w not in model:
            if w[-2] == "'":
                w1 = w.replace("'d", "ed")
                w1 = w1.replace("it's", "it is")
                w1 = w1.replace("'s", "")
            if w1 in model:
                v = my_r2b.to_bin(model[w1])
                return v

        if w in model:
            v = my_r2b.to_bin(model[w])
            return v
        return None

    @staticmethod
    def get_word_vector(fred):
        pass


def get_word2Vec(w):
    '''
    Returns None if word not found in Model, otherwise BSC mapped from model's real number vector
    '''
    w1 = w
    v = None
    if w not in model:
        if w1[0] == "'":
            w1 = w[1:]  # Chop off a leading quote '
        if len(w1) > 0 and w1[-1] == "'":
            w1 = w[:-1]  # Chop off a training quote '
        if len(w1) > 2 and w[-2] == "'":
            w1 = w.replace("'d", "ed")
            w1 = w1.replace("'s", "")

    if w1 in model:
        v = my_r2b.to_bin(model[w1])

    return v


def get_word_vector(w):
    #global model, use_word2vec, include_vecs_for_all_words, skip_words, seen_words
    v = None
    try:
        if use_word2vec:
            v = get_word2Vec(w)

        if v is None:
            v = createWordVector(w).myvec
            if use_word2vec:
                miss_from_word2vec[w] = miss_from_word2vec.get(w, 0) + 1

    except Exception as e:
        e = e

    return v


def get_or_skip_word_vector(w):
    '''
    Returns BSC vector representation of the word or None if failed to build a vector
    (e.g word is in skip list or word contains all non-printing chars.
    '''
    #global model, use_word2vec, include_vecs_for_all_words, skip_words, seen_words
    v = None
    try:
        if not include_vecs_for_all_words and w in skip_words:
            if w not in skip_words:
                skip_words[w] = PackedVec(get_word_vector(w))
            return None

        if w in seen_words:
            if w in miss_from_word2vec:
                miss_from_word2vec[w] += 1  # Count the occurrences of words missing from our word2vec model
            return seen_words[w].myvec
        else:
            v = get_word_vector(w)
            if v is not None:
                seen_words[w] = PackedVec(v)  # We only build new word vectors once.

    except [ValueError, IndexError] as e:
        e = e

    return v


def chunkSentenceVector(sentence1):
    #global model, use_word2vec, include_vecs_for_all_words, skip_words, seen_words
    global total_word_count

    db_sentence = None
    wordvecs = []
    try:
        sentence = sentence1.replace('--', ' ')
        sentence = replace_non_alphanumerics(sentence, ' ')
        # sentence = sentence.translate(strtab, string.punctuation)
        words = sentence.split()

        for w in words:
            w = w.strip()
            w = w.replace("'d", "ed")
            if w is None: continue

            vec = get_or_skip_word_vector(w)
            if vec is not None:
                total_word_count += 1
                wordvecs.append(CSPvec(w, [vec], -1, role_vecs))

        if len(wordvecs) == 0:
            wordvecs = wordvecs
            return None
        else:
            try:
                db_sentence = CSPvec.buildchunks(sentence, wordvecs, hierarchy_level, role_vecs,
                                                 split_tail_evenly=True, rebuild_names=True)
            except [ValueError, IndexError] as e:
                print(e)
                db_sentence = db_sentence

    except [ValueError, IndexError] as e:
        print(e)
        e = e

    return db_sentence

def getWordList(sentence):
    s = sentence.split()
    words = []
    for w in s:
        if w.lower() not in stopWords:
            words.append(w)

    return words


def getHamletActNumber(actstr):
    # Extract the act number form the act descriptor
    actname = actstr.strip().split(',')[0]

    anum = actname.split('Act')[1]
    return int(anum), actname


def getHamletSceneNumber(actstr):
    # Extract the act number form the act descriptor
    scenname = actstr.strip().split(',')[1]
    anum = scenname.split('Scene')[1]
    return int(anum), scenname


def buildactdicts(filename):
    m_cols = ['actor', 'old', 'new']
    df = pd.read_csv(filename, sep='@', names=m_cols, encoding='utf-16')

    currentact = 0
    currentscene = 0
    actdict_old = []
    scenedict_old = []
    actdict_new = []
    scenedict_new = []
    acts_old = []
    scenes_old = []
    lines_old = []
    lines_new = []
    actname = ""
    scenename = ""
    fullactorscombined = defaultdict(lambda: ([], []))
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for index, row in df.iterrows():
        #print(f"index {index}, row {row} ")
        index = index.strip()
        if index == '' or pd.isnull(index):
            actor = row['actor'].replace('"', '').strip()
            oldline = fixup_chars(row['old'])
            newline = fixup_chars(row['new'])
            p = tokenizer.tokenize(oldline)
            lines_old.append((actor, oldline))
            lines_new.append((actor, newline))
            fullactorscombined[actor][0].append(oldline)
            fullactorscombined[actor][1].append(newline)
            # print("\t", actor, oldline, newline)
        else:
            # Here we have an act or a scene change
            if 'END PLAY' in index:
                # Scene is finished
                scenedict_old.append((scenename, lines_old))
                scenedict_new.append((scenename, lines_new))
                # Act is finished
                actdict_old.append((actname, scenedict_old))
                actdict_new.append((actname, scenedict_new))
                break

            actnum, tmpname = getHamletActNumber(index)
            if currentact != actnum:
                if currentact == 0:
                    # We just initialise stuff
                    currentact = actnum
                    actname = tmpname
                    currentscene, scenename = getHamletSceneNumber(index)
                    lines_old = []
                    lines_new = []
                else:
                    # We save the collected act and scene data
                    # Scene is finished
                    scenedict_old.append((scenename, lines_old))
                    scenedict_new.append((scenename, lines_new))
                    # Act is finished
                    actdict_old.append((actname, scenedict_old))
                    actdict_new.append((actname, scenedict_new))
                    # We have a new ACT
                    currentact = actnum
                    actname = tmpname
                    currentscene, scenename = getHamletSceneNumber(index)
                    scenedict_old = []
                    scenedict_new = []
                    lines_old = []
                    lines_new = []
            else:
                # Only the scene number must havechanged

                # Now get the scene number
                tmpnum, tmpname = getHamletSceneNumber(index)
                if tmpnum <= currentscene:
                    # This is an error we expect the scene number to have changed
                    print("Error What happened to the scene number", currentscene, tmpnum)
                else:
                    scenedict_old.append((scenename, lines_old))
                    scenedict_new.append((scenename, lines_new))
                    lines_old = []
                    lines_new = []
                    scenename = tmpname
                    currentscene = tmpnum

    return actdict_old, actdict_new  # , fullactorscombined


def buildacts_from_csv(actdict):
    global linecheck
    vtab = '\t'
    actorDict = {}
    linechunks = []
    subScenes = []
    scenes = []
    acts = []
    scene_cnt = no_scenes_per_act
    act_cnt = no_acts

    linecheck = []
    lcnt = 1
    hierarchy_level = [0]  # This is used to maintain hierarchy level

    for act_name, act in actdict:
        this_act = get_act_number(act_name)
        act_name = act_name.replace("\n", "").strip()
        print("\n\n", act_name)
        # lcnt = 1
        scenes = []
        for scene_name, scene in act:
            this_scene = get_scene_number(scene_name)
            scene_name = scene_name.replace("\n", "").strip()
            print("\n\t", scene_name)
            # lcnt = 1
            subScenes = []
            lineVecs = []
            linechunks = []
            hierarchy_level = [0]  # Lines are at level 0, ATM we need this to get an 'action' service to terminate
            sid = -1
            for actor, detail in scene:
                sid += 1

                l = detail
                if len(l) > 0:
                    if report_input_lines:
                        print("\t\t{:d} {:d} {:s}: {:s}".format(lcnt, sid, actor, l))

                    if not 'End of Scene' in l:
                        lcnt += 1
                        # v = chunkSentenceVector(l)
                        v = chunkSentenceVector(l)
                        # v = chunk_sentence(l)
                        if v is None:
                            log.warn('buildacts_from_nltk_xml: Vector Returned is NONE for line=>{}<'.format(l))
                        else:
                            lineVecs.append(v)

            thisscene = CSPvec.buildchunks(act_name + "_" + scene_name, lineVecs, hierarchy_level, role_vecs)
            scenes.append(thisscene)
            scene_cnt -= 1
            if scene_cnt == 0:
                scene_cnt = no_scenes_per_act
                break  # to test just one scene
        # scenes combine to make acts
        thisact = CSPvec.buildchunks(act_name, scenes, hierarchy_level, role_vecs)
        acts.append(thisact)
        act_cnt -= 1
        if act_cnt == 0:
            act_cnt = no_scenes_per_act
            break  # to test just one scene
    return acts, scenes, linecheck


def buildacts_from_json(_play):

    if isinstance(_play, dict):
        play = _play
    else:
        infile = open(_play, 'r')
        play = json.load(infile)

    global linecheck
    vtab = '\t'
    actorDict = {}
    linechunks = []
    subScenes = []
    scenes = []
    acts = []
    scene_cnt = no_scenes_per_act
    act_cnt = no_acts

    linecheck = []
    lcnt = 1
    hierarchy_level = [0]  # This is used to maintain hierarchy level

    for act_name, act in sorted(iteritems(play)):
        this_act = get_act_number(act_name)
        act_name = act_name.replace("\n", "").strip()
        print("\n\n", act_name)
        # lcnt = 1
        if 1:
            scenes = []
            for scene_name, scene in sorted(iteritems(act)):
                scene_name = scene_name.replace("\n", "").strip()
                print("\n\t", scene_name)
                lineVecs = []
                hierarchy_level = [0]  # Lines are at level 0, ATM we need this to get an 'action' service to terminate
                for sid, detail in sorted(iteritems(scene), key=lambda xyz: get_key(xyz[0])):
                    lines = detail['line'].replace('--', '-')  # we don't want to keep '--' within the stanza
                    lines = fixup_chars(lines)
                    l = lines
                    if len(l) > 0:
                        if report_input_lines:
                            log.info("\t\t{:d} {:s} {:s}: {:s}".format(lcnt, sid, detail['actor'], l))

                        if not 'End of Scene' in l:
                            lcnt += 1
                            if 'How now, Horatio! You tremble and look pale:' in l:
                                l=l
                            v = chunkSentenceVector(l)
                            if v is None:
                                log.warn('buildacts_from_nltk_xml: Vector Returned is NONE for line=>{}<'.format(l))
                                chunkSentenceVector(l)
                            else:
                                lineVecs.append(v)

                    # Debugging
                    # if lcnt > 3:
                    #    break

                thisscene = CSPvec.buildchunks(act_name + "_" + scene_name, lineVecs, hierarchy_level, role_vecs)
                scenes.append(thisscene)
                scene_cnt -= 1
                if scene_cnt == 0:
                    scene_cnt = no_scenes_per_act
                    break  # to test just one scene

        # scenes combine to make acts
        thisact = CSPvec.buildchunks(act_name, scenes, hierarchy_level, role_vecs)
        acts.append(thisact)
        act_cnt -= 1
        if act_cnt == 0:
            act_cnt = no_scenes_per_act
            break  # to test just one scene

    return acts, scenes, linecheck


def buildacts_from_nltk_xml(fname):
    global linecheck
    vtab = '\t'
    actorDict = {}
    linechunks = []
    subScenes = []
    scenes = []
    acts = []
    scene_cnt = no_scenes_per_act
    act_cnt = no_acts

    linecheck = []
    lcnt = 1
    hierarchy_level = [0]  # This is used to maintain hierarchy level

    with open(fname, 'rb') as myfile:
        d = xml.parse(myfile)

    for a in d['PLAY']['ACT']:
        act_name = a['TITLE']
        this_act = get_act_number(act_name)
        print("\n\n", act_name)
        # lcnt = 1
        if 1:
            scenes = []
            for s in a['SCENE']:
                scene_name = s['TITLE']
                this_scene = get_scene_number(scene_name)
                scene_name = scene_name.replace("\n", "").strip()
                print("\n\t", scene_name)
                # lcnt = 1
                subScenes = []
                lineVecs = []
                linechunks = []
                hierarchy_level = [0]  # Lines are at level 0, ATM we need this to get an 'action' service to terminate
                sid = -1
                for p in s['SPEECH']:
                    sid += 1
                    if isinstance(p['LINE'], list):
                        # print('\t\t', p['SPEAKER'])
                        # lines = " ".join([x for x in p['LINE'] if isinstance(x, unicode) or isinstance(x, str)])
                        lines = " ".join([x for x in p['LINE'] if not isinstance(x, xml.OrderedDict)])
                        # print('\t\t\t', l)
                    else:
                        if isinstance(p['LINE'], xml.OrderedDict):
                            # lines=lines
                            continue
                        else:
                            lines = p['LINE']
                        # print('\t\t Z', p['SPEAKER'], p['LINE'])
                    lines = fixup_chars(lines)
                    l = lines
                    if len(l) > 0:
                        if report_input_lines:
                            # log.info("\t\t{:d} {:s}: {:s}".format(lcnt, p['SPEAKER'], l))
                            if 'O horror, horror, horror! Tongue nor heart Cannot conceive nor name thee!' in l:
                                l = l
                            log.info("\t\t{:d} {:s} {:s}: {:s}".format(lcnt, str(sid),
                                                                       ', '.join(p['SPEAKER']) if isinstance(
                                                                           p['SPEAKER'], list) else p['SPEAKER'], l))

                        if not 'End of Scene' in l:
                            lcnt += 1
                            #v = chunkSentenceVector(l)
                            v = chunkSentenceVector(l)
                            if v is None:
                                log.warn('buildacts_from_nltk_xml: Vector Returned is NONE for line=>{}<'.format(l))
                            else:
                                lineVecs.append(v)

                    # Debugging
                    # if lcnt > 3:
                    #    break

                thisscene = CSPvec.buildchunks(act_name + "_" + scene_name, lineVecs, hierarchy_level, role_vecs)
                scenes.append(thisscene)
                scene_cnt -= 1
                if scene_cnt == 0:
                    scene_cnt = no_scenes_per_act
                    break  # to test just one scene

        # scenes combine to make acts
        thisact = CSPvec.buildchunks(act_name, scenes, hierarchy_level, role_vecs)
        acts.append(thisact)
        act_cnt -= 1
        if act_cnt == 0:
            act_cnt = no_scenes_per_act
            break  # to test just one scene

    return acts, scenes, linecheck


role_vecs = RoleVecs(10000)  # Role vecs definitions
if __name__ == '__main__':
    raw_input23 = vars(__builtins__).get('raw_input', input)

    # Set up reporting options and what level of chunking we want, words or sentences

    log = setuplogs(level='INFO')
    # log = setuplogs(level='DEBUGV')
    # log = setuplogs(level='DEBUG')

    random.seed()
    report_input_lines = True
    report_end_of_subtask = True
    show_multiple_matches = False
    usechunksforWords = True
    usechunksforWords_output_individual_words = False
    usechunksforWords_show_word_matches = False
    no_scenes_per_act = 100
    no_acts = 100
    simulate_run = True
    use_separate_actors = False
    use_word2vec = False
    include_vecs_for_all_words = True #lambda w: len(w) > 4
    repeat_test = False

    # Set up the run parameters
    print("Run config:")
    if usechunksforWords:
        print("\n\t\tChunking at 'word' level:")
    else:
        print("\n\t\tChunking at 'line' level:")
    if show_multiple_matches:
        print("\n\t\tShowing multiple matches")
        if usechunksforWords_show_word_matches:
            print("\n\t\tShowing multiple match word break down (verbose)")
        else:
            print("\n\t\tNOT Showing word match break down")
        if usechunksforWords_output_individual_words:
            print("\n\t\tShow chosen word matches on separate lines (verbose)")
        else:
            print("\n\t\tDo NOT show chosen Word matches separately")
    else:
        print("\n\t\tNOT Showing multiple matches")

    if no_acts > 7:
        no_acts_str = "all"
    else:
        no_acts_str = str(no_acts)

    if no_scenes_per_act > 7:
        no_scene_str = "all"
    else:
        no_scene_str = str(no_scenes_per_act)

    print("\n\t\tProcessing", no_acts_str, "acts", no_scene_str, "scenes per act")

    if use_word2vec:
        print('\n\t\tUsing word2vec')
    else:
        print('\n\t\tword2vec NOT used')

    if repeat_test:
        repeat_str = "on"
    else:
        repeat_str = "off"
    print("\n\t\tLooping (to test with different randoms) is", repeat_str)

    # see if we wanna change the run
    inp = raw_input23("\n\nDo you want to change the run setup, Y/N? (default='N'): ")
    if 'y' in inp.lower():
        inp = raw_input23("\n\nChunk to line or word level, L/W? (default='L'): ")
        usechunksforWords = 'w' in inp.lower()

        inp = raw_input23("Show input lines Y/N? (default='N'): ")
        report_input_lines = 'y' in inp.lower()

        inp = raw_input23("Enter number of acts to process (default=''(all)): ")
        if inp != "":
            no_acts = int(inp)
        else:
            no_acts = 1000

        inp = raw_input23("Enter number of scenes per act to process (default=''(all)): ")
        if inp != "":
            no_scenes_per_act = int(inp)
        else:
            no_scenes_per_act = 1000

        # if use_word2vec:
        #     inp = raw_input23("Use word2vec Y/N? (default='Y'): ")
        # else:
        #     inp = raw_input23("Use word2vec Y/N? (default='N'): ")
        # if inp != "":
        #     use_word2vec = 'y' in inp.lower()


    CSPvec.use_shaping = False  # Shaping is not implmented in this demo
    word_format = ''
    word_format += '_CW-T' if usechunksforWords else 'CW-F'
    word_format += '_WV-T' if use_word2vec else '_WV-F'
    word_format += '_PVECS' if usechunksforWords else '_CHAIN'
    word_format += '_SHAPED' if CSPvec.use_shaping else '_NoShape'

    model = None

    symbols = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.;:,_'!?-[]&*()>"
    symbol_dict = createSymbolVectors(symbols, 10000)
    num_dict = create_base_vecs("0", "9", 10000, True)


    skip_words = {}
    # skip_words = {'a', 'it', 'is', 'the'}
    # skip_words_file = 'skip_Words' + word_format + '.bin'
    # skip_words = deserialise_object(skip_words_file, {})

    skip_words['a'] = PackedVec(symbol_dict['a'])
    skip_words['A'] = PackedVec(symbol_dict['A'])
    skip_words['I'] = PackedVec(symbol_dict['I'])
    skip_words['O'] = PackedVec(symbol_dict['O'])
    # # skip_words['to'] = createWordVector('to')
    # skip_words['of'] = createWordVector('of')
    # skip_words['the'] = createWordVector('of')

    miss_from_word2vec = {}
    total_word_count = 0
    seen_words = {}
    # seen_words_file = 'seen_words' + word_format + '.bin'
    # seen_words = deserialise_object(seen_words_file, {})


    if False and use_word2vec:  # Word2vec is not implemented in this demo
        w2vload_startTime = timeit.default_timer()
        model = Word2Vec.load('old_new_english.model').wv
        print('\nLoading word2vec, please wait....')
        # model = KeyedVectors.load_word2vec_format('../Grahams-GensimBinVectors/GoogleNews-vectors-negative300.bin.gz',
        #                                           binary=True)

        print("word2vec loaded - Time taken: {}".format(timeit.default_timer() - w2vload_startTime))

    # Set up what level chunking we want, words or sentences
    if usechunksforWords:
        createWordVector = chunkWordVector  # Using our own chunking scheme, allows individual words as services
    else:
        createWordVector = gb_wordvector_as_chunk  # Graham's piBinding - faster

    startTime = timeit.default_timer()

    vtab = '\t'
    actorDict = {}
    linechunks = []
    subScenes = []
    scenes = []
    acts = []
    scene_cnt = no_scenes_per_act
    act_cnt = no_acts

    linecheck = []
    lcnt = 1
    hierarchy_level = [0]  # This is used to maintain hierarchy level

    actdict_old, actdict_new = buildactdicts('data/source_files/Hamlet_CSV01_UTF16_@Sep.txt')  # Preprocess NOFEAR shakespear versions
    file_path = './data/output_vec_files'

    # -----------------------------------------------------------------------------------------------
    # I have disabled the nltk versions untill I write up how to install nltk and its components.
    #
    # runlist = [('macbeth', '/home/chris/nltk_data/corpora/shakespeare/macbeth.xml', f'{file_path}/macbeth_nltk_{word_format}.bin', buildacts_from_nltk_xml),
    #            ('ntlk_shakespear', '/home/chris/nltk_data/corpora/shakespeare/hamlet.xml', f'{file_path}/hamlet_nltk_{word_format}.bin', buildacts_from_nltk_xml),
    #            ('hamlet_orig', 'data/source_files/hamlet_stanzas.json', f'{file_path}/hamlet_stanzas_{word_format}.bin', buildacts_from_json),
    #            ('nofear_old', actdict_old, f'{file_path}/hamlet_old_{word_format}.bin', buildacts_from_csv),
    #            ('nofear_new', actdict_new, f'{file_path}/hamlet_new_{word_format}.bin', buildacts_from_csv)]
    #
    # I have disabled the nltk versions untill I write up how to install nltk and its components.
    # -----------------------------------------------------------------------------------------------
    runlist = [('hamlet_orig', 'data/source_files/hamlet_stanzas.json', f'{file_path}/hamlet_stanzas_{word_format}.bin', buildacts_from_json),
               ('nofear_old', actdict_old, f'{file_path}/hamlet_old_{word_format}.bin', buildacts_from_csv),
               ('nofear_new', actdict_new, f'{file_path}/hamlet_new_{word_format}.bin', buildacts_from_csv)]


    # my_r2b = Real2Binary(300, 10000, seed=951753)
    docs = []
    run_log = []
    for kk, infn, outfn, parse_func in runlist:
        startTime = timeit.default_timer()
        try:
            total_word_count = 0
            miss_from_word2vec.clear()
            linecheck.clear()
            hierarchy_level = [0]
            acts, scenes, linecheck = parse_func(infn)
            msg = f"{kk}:Total word count={total_word_count}, unique word count={len(seen_words)}, words missing from word2vec model={len(miss_from_word2vec)}"
            run_log.append(msg)
            print(msg)
            print(miss_from_word2vec)
        except Exception as e:
            print(f"******ERROR: {e}")

        top_chunk = CSPvec.buildchunks(kk, acts, hierarchy_level, role_vecs)
        if not os.path.exists(file_path):  # Make sure the output directory exists.
            os.makedirs(file_path)  # Create output directory if not exists.
        serialise_vec_heirarchy(top_chunk, outfn)
        print(f"\n\n{kk.upper()} Time taken to build representation: {timeit.default_timer() - startTime}\n\n")
        del top_chunk  # Free up memory
        del acts
        del scenes

    for m in run_log:
        print(m)
