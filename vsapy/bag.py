import datetime
from enum import IntEnum
import numpy as np
import vsapy as vsa
from .vsatype import VsaBase, VsaType


class TimeStamp(object):
    @staticmethod
    def get_creation_data_time_stamp():
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S.%f}'.format(datetime.datetime.now())

    @staticmethod
    def compare_time_stamps(t1, t2):
        t1obj = datetime.datetime.strptime(t1, 'Timestamp: %Y-%m-%d %H:%M:%S.%f')
        t2obj = datetime.datetime.strptime(t2, 'Timestamp: %Y-%m-%d %H:%M:%S.%f')
        return t1obj.time() == t2obj.time()


def createSymbolVectors(symbols, veclen, creation_data_time_stamp=None):
    if creation_data_time_stamp is None:
        creation_data_time_stamp = TimeStamp.get_creation_data_time_stamp()
    # A dictionary is slightly faster than Graham's list look up but only marginally....
    sym = {"creation_data_time_stamp": creation_data_time_stamp}

    for a in symbols:
        sym[a] = vsa.randvec(veclen)

    return sym


def create_base_vecs(start, end, veclen, time_stamp, ascii_names=True, creation_data_time_stamp = None):
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
        base_vecs.update({c: vsa.randvec(veclen)})
        list(base_vecs.items())

    return base_vecs


class PackedVec(object):
    '''
    Used to reduce memory storage overhead by keeping vectors packed.
    '''
    def __init__(self, v):
        self.myvec = v
        self.vsa_type = v.vsa_type

    @property
    def myvec(self):
        return VsaBase(np.unpackbits(self.__myvec), self.vsa_type)

    @myvec.setter
    def myvec(self, v):
        self.__myvec = np.packbits(v)

    @property
    def packed(self):
        return self.__myvec


class BagVec(PackedVec):
    def __init__(self, veclist, vec_cnt=-1):
        rawvec, vec_cnt, norm_vec = BagVec.bundle(veclist, vec_cnt)
        super(BagVec, self).__init__(norm_vec)
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
                return veclist, vec_cnt,  vsa.normalize(veclist, vec_cnt, )
        elif isinstance(veclist, list):    # i.e. only 1D array
            if not isinstance(veclist[0], list):
                assert vec_cnt >= 1, "you must specify parameter 'vec_cnt': number of vectors in this vector."
                return veclist, vec_cnt,  vsa.normalize(veclist, vec_cnt, )
        else:
            raise ValueError(" 'veclist' is not an array type.")

        rawvec = np.sum(veclist, axis=0)
        if vec_cnt >= 1:
            # This enables passing in one or more un-normalized vectors in veclist. If vec_cnt >= 1 we assume that
            # vec_cnt accounts for all un-normalised vectors in the list
            norm_vec = vsa.normalize(rawvec, vec_cnt, )
        else:
            norm_vec = vsa.normalize(rawvec, len(veclist), )
            vec_cnt = len(veclist)

        return rawvec, vec_cnt, norm_vec


class RawVec(BagVec):
    def __init__(self, veclist, vec_cnt=-1):
        rawvec, vec_cnt, norm_vec = RawVec.bundle(veclist, vec_cnt)
        super(RawVec, self).__init__(rawvec, vec_cnt)
        self.__rawvec = rawvec

    @property
    def rawvec(self):
        return self.__rawvec

    @rawvec.setter
    def rawvec(self, rawvec):
        self.__rawvec = rawvec


class BareChunk(BagVec):
    def __init__(self, cnk):
        super(BareChunk, self).__init__(cnk.rawvec, cnk.vec_cnt)
        self.aname = cnk.aname
        self.chunklist = [BareChunk(c) for c in cnk.chunklist] if not cnk.isTerminalNode else []


    @property
    def isTerminalNode(self):
        return self.chunklist is None

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

    def get_level_number(self):
        return self.aname[self.aname.find("$")+1:self.aname.find("@")]

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
