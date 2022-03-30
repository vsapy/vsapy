import os
from os.path import exists
import pickle
import datetime
from enum import IntEnum
import numpy as np
import vsapy.vsapy as vsa
from vsapy.vsatype import *
from vsapy.helpers import serialise_object


class TimeStamp(object):
    @staticmethod
    def get_creation_data_time_stamp():
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S.%f}'.format(datetime.datetime.now())

    @staticmethod
    def compare_time_stamps(t1, t2):
        t1obj = datetime.datetime.strptime(t1, 'Timestamp: %Y-%m-%d %H:%M:%S.%f')
        t2obj = datetime.datetime.strptime(t2, 'Timestamp: %Y-%m-%d %H:%M:%S.%f')
        return t1obj.time() == t2obj.time()


def createSymbolVectors(symbols, *args, creation_data_time_stamp=None, **kwargs):
    if creation_data_time_stamp is None:
        creation_data_time_stamp = TimeStamp.get_creation_data_time_stamp()
    # A dictionary is slightly faster than Graham's list look up but only marginally....
    sym = {"creation_data_time_stamp": creation_data_time_stamp}

    for a in symbols:
        sym[a] = vsa.randvec(*args, **kwargs)

    return sym


def create_base_vecs(start, end, veclen, ascii_names=True,
                     creation_data_time_stamp=None, vsa_type=VsaType.BSC, **kwargs):

    if creation_data_time_stamp is None:
        time_stamp = TimeStamp.get_creation_data_time_stamp()
    else:
        time_stamp = creation_data_time_stamp

    base_vecs = {"creation_data_time_stamp": time_stamp}  # Dictionary(Of String, BitArray)
    if ascii_names:
        start = ord(start)
        end = ord(end)
    for x in range(start, end + 1):  # inclusive of end
        if ascii_names:
            c = chr(x)
        else:
            c = x
        base_vecs.update({c: vsa.randvec(veclen, vsa_type=vsa_type, **kwargs)})
        list(base_vecs.items())

    return base_vecs


class PackedVec(object):
    '''
    Used to reduce memory storage overhead by keeping vectors packed.
    '''
    def __init__(self, v):
        self.vsa_type = v.vsa_type
        self.myvec = v

    @property
    def myvec(self):
        # performance for Laiho/X is slightly improved by not searching the subclass chain
        if self.vsa_type == VsaType.Laiho or self.vsa_type == VsaType.LaihoX:
            return self.__myvec
        else:
            return VsaBase.get_subclass(self.vsa_type).unpackbits(self.__myvec)

    @myvec.setter
    def myvec(self, v):
        # performance for Laiho/X is slightly improved by not searching the subclass chain
        if isinstance(v, vsa.Laiho):
            self.__myvec = v
        else:
            self.__myvec = v.packbits(v)

    @property
    def packed(self):
        return self.__myvec


class BagVec(PackedVec):
    def __init__(self, veclist, vec_cnt=-1, myvec=None):
        """

        :param veclist: list of vectors to be majority-summed.
        :param vec_cnt: If the vector is already summed the number of vectors in the sum. This allows a mix of
                        normalised and un-normalised vectors to be passed in veclist. For example,
                        veclist = [[2,1,3,5,0], [1,0,1,1,0,1]] then vec_cnt might = 7
                        because 1st vec could be an un-normallised sum of 6 vecs and the second vec is normalised.
        """
        if myvec is None:
            rawvec, vec_cnt, myvec = BagVec.bundle(veclist, vec_cnt)
        else:
            assert vec_cnt >= 1, "you must specify parameter 'vec_cnt': number of vectors in this vector."

        super(BagVec, self).__init__(myvec)
        self.vec_cnt = vec_cnt

    @property
    def vec_cnt(self):
        return self.__vec_cnt

    @vec_cnt.setter
    def vec_cnt(self, vcount):
        self.__vec_cnt = vcount

    @staticmethod
    def bundle(veclist, vec_cnt=-1):
        if isinstance(veclist, np.ndarray):
            if len(veclist.shape) == 1:   # i.e. only 1D array
                assert vec_cnt >= 1, "you must specify parameter 'vec_cnt': number of vectors in this vector."
                rawvec = veclist  # veclist is a single [a1, a2, a3, ...] numpy column vector
                return rawvec, vec_cnt,  vsa.normalize(veclist, vec_cnt, )
        elif isinstance(veclist, list):
            if not isinstance(veclist[0], (list, np.ndarray)):  # i.e. only 1D array
                assert vec_cnt >= 1, "you must specify parameter 'vec_cnt': number of vectors in this vector."
                rawvec = veclist  # veclist is a single [a1, a2, a3, ...] python list vector
                return rawvec, vec_cnt,  vsa.normalize(veclist, vec_cnt, )
        else:
            raise ValueError(" 'veclist' is not an array type.")

        # veclist contains a list of vectors t be added.
        sumv = veclist[0].sum(veclist)
        if isinstance(sumv, vsa.Laiho):
            rawvec = None
            norm_vec = sumv
            vec_cnt = len(veclist)
        else:
            rawvec = sumv
            if vec_cnt >= 1:
                # This enables passing in one or more un-normalized vectors in veclist. If vec_cnt >= 1 we assume that
                # vec_cnt accounts for all un-normalised vectors in the list
                norm_vec = vsa.normalize(sumv, vec_cnt, )
            else:
                norm_vec = vsa.normalize(rawvec, len(veclist), )
                vec_cnt = len(veclist)

        return rawvec, vec_cnt, norm_vec


class RawVec(BagVec):
    def __init__(self, veclist, vec_cnt=-1):
        rawvec, vec_cnt, myvec = BagVec.bundle(veclist, vec_cnt)
        super(RawVec, self).__init__(None, vec_cnt, myvec=myvec)

        if isinstance(veclist[0], (np.ndarray, list)) and isinstance(veclist[0], vsa.Laiho):
            self.__rawvec = self.myvec
        else:
            self.__rawvec = rawvec

    @property
    def rawvec(self):
        return self.__rawvec

    @rawvec.setter
    def rawvec(self, rawvec):
        self.__rawvec = rawvec


def serialise_vec_hierarchy(chunk_hierarchy, pathfn):
    bare_hamlet = BareChunk(chunk_hierarchy)
    serialise_object(bare_hamlet, pathfn)
    return


class BareChunk(RawVec):
    def __init__(self, cnk):
        super(BareChunk, self).__init__(cnk.rawvec, cnk.vec_cnt)
        self.vsa_type = cnk.vsa_type
        self.creation_data_time_stamp = cnk.creation_data_time_stamp
        self.aname = cnk.aname
        self.chunklist = [BareChunk(c) for c in cnk.chunklist] if not cnk.isTerminalNode else []
        self.__terminal_node = cnk.isTerminalNode

    @property
    def isTerminalNode(self):
        return self.__terminal_node

    def get_acts(self, name_prefix=''):
        if name_prefix != '':
            acts = []
            for act in self.chunklist:
                act.aname = f'{name_prefix}@{act.aname}'
                acts.append(act)
            return acts
        else:
            return [act for act in self.chunklist]

    def flattenchunkheirachy(self, allchunks, skip_worker_chunks=False):
        if self.isTerminalNode:
            if skip_worker_chunks:
                return
            else:
                allchunks.append(self)
            return
        else:
            allchunks.append(self)
            for c in self.chunklist:
                c.flattenchunkheirachy(allchunks, skip_worker_chunks)

    @staticmethod
    def get_level_number(aname):
        return aname[aname.find("$")+1:aname.find("@")]

    @staticmethod
    def add_level_labels(cnk, current_level, doc_prefix, label=''):
        if current_level == 0 or label == '':
            label = '00'
        else:
            label = f"{label}-{current_level:02d}"

        cnk.aname = f"{doc_prefix}${label}@{cnk.aname}"
        if not cnk.isTerminalNode:
            current_level = 1
            for c in cnk.chunklist:
                BareChunk.add_level_labels(c, current_level, doc_prefix, label)
                current_level += 1

        return

    @staticmethod
    def get_levels_as_list(cnk, levels, current_level_request, cnk_list):
        """
        Note this method retrieves chunks directly from a stored heirarchical chunk tree - NOT by using unbinding.
        :param cnk: top level chunk from which to start the retrieval
        :param levels:  pass a list of the levels you want to retrieve, in this case [0, 1] = the whole play and each
                act. Passing [2] would retrieve only the scene data.
        :param current_level_request: recursive leval tracker passing zero to start should always work
        :param cnk_list: flat list of retrieved chunks. Passed as parameter because of recursion.
        """
        ll = levels[:]
        ll.sort()  # Levels need to be in order for retrieval to work.
        if current_level_request == ll[0]:
            cnk_list.append(cnk)
            ll = ll[1:]  # pinch off the first element

        # descend cnk hierarchy until we are at level zero
        if len(ll) > 0:
            current_level_request += 1
            for c in cnk.chunklist:
                BareChunk.get_levels_as_list(c, ll, current_level_request, cnk_list)

        return

class BareChunkRunDetail(object):
    '''
    Saves a chunk hierarchy with details of how it was built, e.g.
            input
            encoding scheme for leaf nodes:
                    chain (unique vectors for each word),
                    pvec (syntatic matching)
                    WV-T = True, WV-F = False - whether word2vec symantic vectors where used for leaf nodes.
                    CW-T = True, CW-F = False - hierarchy descends to word level for leaf nodes

    '''

    def __init__(self, chunk_hierarchy, run_detail, play_name, short_info, long_info):
        self.play_name = play_name
        self.run_detail = run_detail
        self.short_info = short_info
        self.long_info = long_info
        self.top_chunk = chunk_hierarchy
