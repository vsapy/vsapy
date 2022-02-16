import threading
import datetime
from enum import IntEnum
import numpy as np
import vsapy as vsa
from .vsatype import VsaBase, VsaType
from vsapy.bag import *
from vsapy.role_vectors import *
from vsapy.helpers import deserialise_object

from vsapy.logger_utils import *
log = setuplogs(level='INFO')


class CSPvec(RawVec):
    id_stamp = "NOT SET"
    mythreadlock = threading.RLock()
    next_chunk_id = 0
    break_on_chunk_id = -1
    trace_threshold = 0.53  # 0.547 #0.528  #0.53  # 0.525  # 0.54  # 0.532

    # ------------------------------------------------------------------------
    # If we change this list we MUST update calc_match_level():
    # trace_thresholds = [0.8, 0.7, 0.6, 0.54, 0.53, 0.525]
    # trace_thresholds = [0.6, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.525]
    # trace_thresholds = [0.6, 0.56, 0.54]
    trace_thresholds = [0.53]
    # trace_thresholds = [0.8, 0.7, 0.54]

    def __init__(self, name, veclist, level, role_vecs, maxvecs=32, chunks=None, creation_data_time_stamp=None):

        self.creation_data_time_stamp = role_vecs.creation_data_time_stamp

        # if "NOT SET" in CSPvec.id_stamp:  # This means class vars have never been initialised
        #     CSPvec.id_stamp = self.creation_data_time_stamp

        self.im_the_requester = False  # Service owning this vector is the requester
        self.aname = name
        self.level = level
        self.ref_id = -1
        self.start_offset = -1000
        self.chunksize = len(veclist)
        self.veclen = len(veclist[0])

        self.roles = role_vecs
        self.permVecs = role_vecs.permVecs

        self.chunk_id = CSPvec.next_chunk_id
        with CSPvec.mythreadlock:
            CSPvec.next_chunk_id += 1

        self.chunklist = chunks  # the real vectors, for debug and possible use in clean up
        self.start_tag_vec = None

        # The stopvec is made from the last vector in the list this ensures that we get better matches
        # the main reason for this is to ensure that the last child vector in the list is able
        # to detect that the next vector is a stop vector.
        # The spinoff benefit is that we get better matches because vector lists ending in the same
        # vector will not differ by the stop vector.
        self.stopvec = vsa.bind(self.roles.role_stopvec, np.roll(veclist[-1], 1))
        if self.isTerminalNode:
            # we are creating a basic compound vector these do not need a stop vector, however if the number of vectors
            # to be added is even we add one enyway to make the majority vote work niceley
            if self.chunksize % 2 == 0:
                # For terminal nodes this stop_vec is actually a random vec but lets us keep an exact count
                # of the vectors in the compound vec
                veclist.append(self.stopvec)
        else:
            veclist.append(self.stopvec)


        rawvec, vec_cnt, norm_vec  = self.addvecs(veclist)
        super(CSPvec, self).__init__(rawvec, vec_cnt)

        if self.isTerminalNode:
            # If we are a terminal node we will never look for our own stop_vec, however we do want to detect the
            # parent vectors stop_vec.  We therefore recalc the stop_vec to match with any parent vectors
            # stop_vec should 'this' vec be last in the list.
            self.stopvec = vsa.bind(self.roles.role_stopvec, np.roll(self.myvec, 1))
    

    def addvecs(self, veclist):
        """
        Adding like this, p0 * a + (p0 * p1 ** b) + (p0 * p1 * p2 * c) +
        Using this method we gain benefit because we will get better similarity matchups
        since sequence is controlled by a fixed set of random vectors
        sumvec = self.prep_payload(veclist[0], 1, mypayload)

        :param veclist:
        :return:
        """

        if len(veclist) == 1:
            return veclist[0], 1, veclist[0]

        try:
            pindex = 0
            piv = self.permVecs[0]
            sumvec = vsa.bind(piv, np.roll(veclist[0], pindex + 1))
            cnt = 1
            for y in veclist[1:]:
                cnt += 1
                pindex += 1
                piv = vsa.bind(piv, self.permVecs[pindex])
                v = vsa.bind(piv, np.roll(y, pindex + 1))
                sumvec = sumvec + v

        except IndexError as e:
            pindex = pindex  # For Debug
            raise IndexError(e)

        nv = vsa.normalize(sumvec, cnt)
        return sumvec, cnt, nv

    @property
    def isTerminalNode(self):
        return self.chunklist is None

    @classmethod
    def get_refid_from_tvec_perms(cls, invec):
        for i in range(len(CSPvec.tvec_permutations)):
            if vsa.randvec(invec, CSPvec.tvec_permutations[i]) > 0.53:
                break
        return i

    @classmethod
    def createchunkfromvecs(cls, name, veclist, level, rolevecs, maxvecs=32):
        return cls(name, veclist, level, rolevecs, maxvecs)

    @classmethod
    def createchunkfromchunks(cls, name, chunks, level, rolevecs, maxvecs=32):
        veclist = [c.myvec for c in chunks]
        return cls(name, veclist, level, rolevecs, maxvecs, chunks)

    @classmethod
    def createchunkfromchunks1(cls, name, chunks, level, role_vecs, maxvecs=32):
        name = ' '.join([c.aname for c in chunks])
        veclist = [c.myvec for c in chunks]
        return cls(name, veclist, level, role_vecs, maxvecs, chunks)

    @classmethod
    def buildchunks(cls, name, chunklist, level, role_vecs, split_tail_evenly=True, maxvecs=32, rebuild_names=False):
        """

        :param name: Name of this chunk
        :param chunklist: list of individual vectors
        :param level: hiearchy_level intentional side effect modifies level in outside world
        :param split_tail_evenly: True=tails of list split approx in half, else straglers just moppoed up into one vec
        :return: top level chunk of a heirachy built from the flat list of chunks supplied
        """

        try:
            if len(chunklist) == 0:
                chunklist = chunklist
            if chunklist is None:
                chunklist = chunklist
            if isinstance(chunklist[0], CSPvec):
                if rebuild_names:
                    create_chunk = cls.createchunkfromchunks1
                else:
                    create_chunk = cls.createchunkfromchunks
            else:
                create_chunk = cls.createchunkfromvecs
        except IndexError as e:
            e = e

        m = 0
        llen = len(chunklist)
        if llen > maxvecs:
            while llen > maxvecs:
                chnks = []
                part_id = 0
                m = 0
                while m + maxvecs <= llen:
                    # when the remaining list is less than two chunks we want to split it equally
                    if (llen - m) % maxvecs != 0 and split_tail_evenly and (llen - m < maxvecs * 2):
                        break
                    sub_chunk_list = chunklist[m:m + maxvecs]
                    chnks.append(create_chunk("{}, L{:02d}, part{:02d}".format(name, level[0], part_id),
                                              sub_chunk_list, level[0], role_vecs, maxvecs=maxvecs))
                    m += maxvecs
                    part_id += 1

                if m < llen:  # if there is a tail left to chunk
                    if split_tail_evenly:
                        # we want to split the remainder into two fairly even chunks
                        msplit = int((llen - m) / 2)
                        chnks.append(create_chunk("{}, L{:02d}, part{:02d}".format(name, level[0], part_id),
                                                  chunklist[m:m + msplit], level[0], role_vecs, maxvecs=maxvecs))
                        m += msplit
                        part_id += 1
                        chnks.append(create_chunk("{}, L{:02d}, part{:02d}".format(name, level[0], part_id),
                                                  chunklist[m:llen], level[0], role_vecs, maxvecs=maxvecs))
                    else:
                        # grab any left over vecs
                        chnks.append(create_chunk("{}, L{:02d}, part{:02d}".format(name, level[0], part_id),
                                                  chunklist[m:llen], level[0], role_vecs, maxvecs=maxvecs))

                level[0] += 1  # sub chunks were built so nexy level is at one level higher than its children
                if rebuild_names:
                    create_chunk = cls.createchunkfromchunks1  # after the first pass we are always dealing with chunks
                else:
                    create_chunk = cls.createchunkfromchunks  # after the first pass we are always dealing with chunks

                if len(chnks) <= maxvecs:
                    chunklist = chnks
                    break
                else:
                    llen = len(chnks)
                    chunklist = chnks[:]

        dbcnks = create_chunk(name, chunklist, level[0], role_vecs, maxvecs=maxvecs)
        return dbcnks

    @staticmethod
    def unbind_one_step(invec, pindex):
        return np.roll(vsa.bind(np.roll(CSPvec.permVecs[pindex], -1 * pindex), invec), -1)

    @staticmethod
    def recover_stop_vec_from_vec_count(invec, vec_count):
        invec1 = invec[:]
        for pindex in range(vec_count):
            invec1 = CSPvec.unbind_one_step(invec1, pindex)

        return invec1


    def as_seq(self, seq_posn, im_the_requester=False, target_chan=1, position_request= None):

        self.im_the_requester = im_the_requester

        # We write the total number of sub-vectors into the vector, if the number of sub-vecs in self.myvec_raw
        # is even, then the total number of vecs will be extended by 1 during normalisation.
        vec_count = self.vec_cnt + 2
        if vec_count % 2 == 0:
            vec_count += 1  # An extra vec is added during normalization.

        sub_vecs = [
            self.rawvec,
            self.roles.role_tvec_tag,
            np.roll(self.roles.role_vec_count, vec_count),
        ]

        seq_vec = BagVec(sub_vecs, vec_count)
        start_vec = np.roll(vsa.bind(self.roles.permVecs[0], seq_vec), -1)

        log.infow(f"{self.chunk_id:04d}: START_VEC: NoSubVecs({vec_count}) vec_len={len(start_vec)} | {self.aname}")

        # ChunkService.get_meta_data_debug(start_vec, seq_posn)  # For DEBUG
        return start_vec

    def check_for_activation(self, invec, myvec, required_threshhold):

        if myvec is None:
            return 0, invec, -1

        match = vsa.randvec(invec, myvec)
        # TODO: change check to match the current threshold requiements - this will give us some speed up.
        if match < required_threshhold:
            return match, invec, -1, invec
        else:
            return self.new_get_next_vector(invec, match)
            # next_vec, posn = self.new_get_next_vector(invec, match)
            # return match, next_vec, posn

    def get_current_permutation(self, invec):
        if self.roles.tvec_permutations:
            return self.get_refid_from_tvec_perms(invec)
        pindex = 0
        pvec = self.roles.role_tvec_tag.copy()
        for p in self.roles.permVecs:
            pvec = np.roll(vsa.bind(pvec, np.roll(p, pindex)), -1)
            hd = vsa.randvec(pvec, invec)
            if hd >= CSPvec.trace_threshold:
                # We know if we are better than threshold that we have the best match on role_posnfinder
                # because only one pvec.role_posnfinder combo will ever be a match
                break
            pindex -= 1
        if abs(pindex) >= len(self.roles.permVecs):
            log.error("ERROR: {} Could not find 'T' vector in input vector".format(self.aname))
        # try:
        #     assert abs(pindex) < len(NewChunkPvecs.permVecs), \
        #         "ERROR: {} Could not find 'T' vector in input vector".format(self.aname)
        # except:
        #     pindex = pindex
        return abs(pindex)

    def reverse_unbind(self, invec, debug_prev_vec=None):
        pindex = self.get_current_permutation(invec)
        reverse_vec = np.roll(vsa.bind(np.roll(self.roles.permVecs[pindex], -1 - pindex), invec), 1)
        if debug_prev_vec is not None:
            rhd = vsa.hsim(debug_prev_vec, reverse_vec)
        return reverse_vec, pindex

    def forward_unbind(self, invec):
        try:
            pindex = 0 - self.get_current_permutation(invec) - 1
            nextvec = np.roll(vsa.bind(np.roll(self.roles.permVecs[0 - pindex], pindex), invec), -1)
        except IndexError:
            invec = invec

        return nextvec, pindex

    def wind_vec_forward(self, invec, dest_pindex, debug_target_vec=None):
        invec_pindex = self.get_current_permutation(invec)
        if invec_pindex == dest_pindex:
            return invec

        unbind = self.forward_unbind
        if invec_pindex > dest_pindex:
            unbind = self.reverse_unbind

        nextvec = invec.copy()
        while True:
            nextvec, pindex = unbind(nextvec)
            if abs(pindex) == dest_pindex:
                break

        return nextvec

    def new_get_next_vector(self, invec, hd_match):

        try:
            # pvec = np.roll(NewChunkPvecs.permVecs[0 - pindex], pindex)
            nextvec, pindex = self.forward_unbind(invec)
        except IndexError:
            log.error("{:04d}: {} - Could not resolve Tvec to calc next vec".format(self.chunk_id, self.aname))
            # We return 0 instead of hd_match to indicate that the match was unrelaible
            # and should be discounted as a valid match when check_for_activation() returns
            return 0, invec, -1, invec

        # Test reverse_bind
        # reverse_vec, _ = self.reverse_unbind(nextvec, invec)

        try:
            # pvec = np.roll(NewChunkPvecs.permVecs[0 - ppindex], ppindex)
            nextvec2, _ = self.forward_unbind(nextvec)
        except IndexError:
            log.error("{:04d}: {} - Could not resolve Tvec to calc next vec".format(self.chunk_id, self.aname))
            # We return 0 instead of hd_match to indicate that the match was unrelaible
            # and should be discounted as a valid match when check_for_activation() returns
            return 0, invec, -1, invec

        # nextvec2 = np.roll(vsa.bind(pvec, invec), -1)

        # except IndexError:
        #    pindex=pindex

        return hd_match, nextvec, -1 - pindex, nextvec2

    @staticmethod
    def get_numeric(datavec, role_vec, stop_at):

        i = 0
        for i in range(stop_at):
            if vsa.randvec(np.roll(role_vec, i), datavec) >= CSPvec.trace_threshold:
                break

        return i

    def get_next_vector(self):
        pvec = np.roll(CSPvec[0][self.next_vec + 1], -1 * self.next_vec)
        nextvec = np.roll(vsa.bind(pvec, self.commandvec), -1)
        return nextvec

    def check_for_stopvec(self, invec, myvec):
        return self.check_for_activation(invec, myvec, CSPvec.trace_thresholds[0])  # CS_Fixup01


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

    def get_flat_list_of_vectors_from_chunkheirachy(self, all_vecs):
        # We only want vectors that represent services
        # So we are picking each self.myvec entry from every chunk
        if self.isTerminalNode:
            all_vecs.append(self.myvec)
            return
        else:
            all_vecs.append(self.myvec)
            for c in self.chunklist:
                c.get_flat_list_of_vectors_from_chunkheirachy(all_vecs)

