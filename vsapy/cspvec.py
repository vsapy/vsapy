import vsapy
from vsapy.role_vectors import *

from vsapy.logger_utils import *
log = setuplogs(level='INFO')


class CSPvec(RawVec):
    id_stamp = "NOT SET"
    mythreadlock = threading.RLock()
    next_chunk_id = 0
    break_on_chunk_id = -1
    trace_threshold = 0.013  # 0.53  # 0.547 #0.528  #0.53  # 0.525  # 0.54  # 0.532

    # ------------------------------------------------------------------------
    # If we change this list we MUST update calc_match_level():
    # trace_thresholds = [0.8, 0.7, 0.6, 0.54, 0.53, 0.525]
    # trace_thresholds = [0.6, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.525]
    # trace_thresholds = [0.6, 0.56, 0.54]
    trace_thresholds = [0.53]
    # trace_thresholds = [0.8, 0.7, 0.54]

    def __init__(self, name, veclist, role_vecs, chunks=None, creation_data_time_stamp=None):

        self.creation_data_time_stamp = role_vecs.creation_data_time_stamp

        self.aname = name
        if isinstance(veclist[0], CSPvec):
            # This means we are building from a list of previously created chunks
            self.__terminal_node = False
            self.chunklist = veclist  # the real vectors, for debug and possible use in clean up
            veclist = [c.myvec for c in veclist]
        else:
            self.__terminal_node = True
            # veclist = veclist

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
        self.stopvec = vsa.bind(np.roll(veclist[-1], 1), self.roles.stopvec)

        if self.isTerminalNode:
            # we are creating a basic compound vector these do not need a stop vector, however if the number of vectors
            # to be added is even we add one enyway to make the majority vote work niceley
            if self.chunksize % 2 == 0:
                # For terminal nodes this stop_vec is actually a random vec but lets us keep an exact count
                # of the vectors in the compound vec
                veclist.append(self.stopvec)
        else:
            veclist.append(self.stopvec)

        permed_vec_list = self.permveclist(veclist)
        super(CSPvec, self).__init__(permed_vec_list)

        if self.isTerminalNode:
            # If we are a terminal node we will never look for our own stop_vec. However, we do want to detect the
            # parent vector's stop_vec.  We therefore pre-set the stop_vec to match with self.myvec because, if this
            # vec is in the last position of a higher-level list, self.myvec will be used to build the parent's stopvec.
            self.stopvec = vsa.bind(np.roll(self.myvec, 1), self.roles.stopvec)
    

    def permveclist(self, veclist):
        """
        Adding like this, p0 * a^1 + (p0 * p1 * b^2) + (p0 * p1 * p2 * c^3) + ... Where '^' means cyclic-shift
        Using this method we gain benefit because we will get better similarity matchups
        since sequence is controlled by a fixed set of random vectors

        :param veclist:
        :return: Un-normalised_vec, sub-vec_cnt, normalised-vec
        """

        if len(veclist) == 1:
            return veclist

        try:
            pindex = 0
            piv = self.permVecs[0]
            #sumvec = vsa.bind(piv, np.roll(veclist[0], pindex + 1))
            role_filler_list = [vsa.bind(piv, np.roll(veclist[0], pindex + 1))]
            cnt = 1
            for y in veclist[1:]:
                cnt += 1
                pindex += 1
                piv = vsa.bind(piv, self.permVecs[pindex])
                v = vsa.bind(piv, np.roll(y, pindex + 1))
                role_filler_list.append(v)

        except IndexError as e:
            pindex = pindex  # For Debug
            raise IndexError(e)

        return role_filler_list

    @property
    def isTerminalNode(self):
        return self.__terminal_node

    @classmethod
    def createchunk(cls, name, chunks, rolevecs):
        return cls(name, chunks, rolevecs, chunks)

    @classmethod
    def createchunk_build_name_from_chunks(cls, name, chunks, role_vecs):
        name = ' '.join([c.aname for c in chunks])
        return cls(name, chunks, role_vecs, chunks)

    @classmethod
    def buildchunks(cls, name, chunklist, role_vecs, split_tail_evenly=True, maxvecs=32, rebuild_names=False):
        """

        :param name: Name of this chunk
        :param chunklist: list of individual vectors
        :param split_tail_evenly: True=tails of list split approx in half, else straglers just moppoed up into one vec
        :return: top level chunk of a heirachy built from the flat list of chunks supplied
        """

        assert chunklist is not None, "chunklist is not empty"
        if isinstance(chunklist[0], CSPvec):
            level = 1
        else:
            level = 0

        if rebuild_names:
            create_chunk = cls.createchunk_build_name_from_chunks
        else:
            create_chunk = cls.createchunk

        # m = 0
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
                    chnks.append(create_chunk("{}, L{:02d}, part{:02d}".format(name, level, part_id),
                                              sub_chunk_list, role_vecs))
                    m += maxvecs
                    part_id += 1

                if m < llen:  # if there is a tail left to chunk
                    if split_tail_evenly:
                        # we want to split the remainder into two fairly even chunks
                        msplit = int((llen - m) / 2)
                        chnks.append(create_chunk(f"{name}, L{level:02d}, part{part_id:02d}",
                                                  chunklist[m:m + msplit], role_vecs))
                        m += msplit
                        part_id += 1
                        chnks.append(create_chunk(f"{name}, L{level:02d}, part{part_id:02d}",
                                                  chunklist[m:llen], role_vecs))
                    else:
                        # grab any left over vecs
                        chnks.append(create_chunk(f"{name}, L{level:02d}, part{part_id:02d}",
                                                  chunklist[m:llen], role_vecs,))

                level += 1  # sub chunks were built so next level is at one level higher than its children
                if len(chnks) <= maxvecs:
                    chunklist = chnks
                    break
                else:
                    llen = len(chnks)
                    chunklist = chnks[:]

        dbcnks = create_chunk(name, chunklist, role_vecs)
        return dbcnks

    def unbind_one_step(self, invec, pindex):
        return np.roll(vsa.unbind(invec, np.roll(self.roles.permVecs[pindex], -1 * pindex)), -1)

    def recover_stop_vec_from_vec_count(self, invec, vec_count):
        invec1 = invec[:]
        for pindex in range(vec_count):
            invec1 = self.unbind_one_step(invec1, pindex)

        return invec1

    def as_seq(self, im_the_requester=False):

        self.im_the_requester = im_the_requester

        # We write the total number of sub-vectors into the vector, if the number of sub-vecs in self.vec_cnt
        # is even, then the total number of vecs will be extended by 1 during normalisation.

        vec_count = self.vec_cnt + 2
        if not isinstance(self.myvec, vsa.Laiho):
            if vec_count % 2 == 0:
                vec_count += 1  # An extra vec is added during normalization.
        sub_vecs = [
            self.rawvec,
            self.roles.tvec_tag,  # metadata, identifies the start of a sequence vector
            np.roll(self.roles.vec_count, vec_count),  # metadata, number of sub-vecs inn this vec including metadata
        ]

        bagvec = BagVec(sub_vecs, vec_count)
        if isinstance(self.myvec, vsa.Laiho):
            seq_vec = VsaBase(bagvec.myvec, vsa_type=bagvec.myvec.vsa_type, bits_per_slot=sub_vecs[0].bits_per_slot)
        else:
            seq_vec = VsaBase(bagvec.myvec, vsa_type=bagvec.myvec.vsa_type)

        start_vec = self.unbind_one_step(seq_vec, 0)  # Zero is the start pindex

        log.info(f"{self.chunk_id:04d}: START_VEC: NoSubVecs({vec_count}) vec_len={len(start_vec)} | {self.aname}")

        return start_vec

    def match_theshold(self, invec):
        if isinstance(invec, vsapy.Laiho):
            M = len(invec)

            exprv = 1 / invec.bits_per_slot  # Probability of a match between random vectors
            var_rv = M * (exprv * (1 - exprv))  # Varience (un-normalised)
            std_rv = math.sqrt(var_rv)  # Stdev (un-normalised)
            hdrv = M / invec.bits_per_slot + 4.4 * std_rv  # Un-normalised hsim of two randomvectors adjusted by 'n' stdevs

            # expected_hd = 1 / invec.bits_per_slot  # Normalised expeced value
            # stdev = math.sqrt(len(invec) * expected_hd * (1 - expected_hd)) / len(invec)
            return hdrv/M
        elif invec.vsa_type == VsaType.HRR:
            stdev = math.sqrt(1 / len(invec) * (101 / 100))  # Assuming max 100 vectors
            return 0 + 4.4 * stdev  # mean=0 + 4.4 * stdev
        else:
            return 0.53

    def check_for_activation(self, invec, myvec, required_threshhold):

        if myvec is None:
            return 0, invec, -1

        match = vsa.hsim(invec, myvec)
        # TODO: change check to match the current threshold requiements - this will give us some speed up.
        if match < required_threshhold:
            return match, invec, -1, invec
        else:
            return self.new_get_next_vector(invec, match)

    def get_current_permutation(self, invec):
        if self.check_for_start_tag_vec(invec):
            return -1
        #match_threshold = self.match_theshold(invec)
        #match_threshold = VsaBase.random_threshold(invec)
        match_threshold = vsa.random_threshold(invec)

        pindex = 0
        pvec = self.roles.tvec_tag.copy()
        for p in self.roles.permVecs:
            pvec = self.unbind_one_step(pvec, abs(pindex))
            hd = vsa.hsim(pvec, invec)
            if hd >= match_threshold:
                # We know if we are better than threshold that we have the best match on role_posn finder
                # because only one pvec.role_posn finder combo will ever be a match
                break
            pindex -= 1
        if abs(pindex) >= len(self.roles.permVecs):
            log.error("ERROR: {} Could not find 'T' vector in input vector".format(self.aname))

        return abs(pindex)

    def reverse_unbind(self, invec, debug_prev_vec=None):
        pindex = self.get_current_permutation(invec)
        reverse_vec = np.roll(vsa.bind(invec, np.roll(self.roles.permVecs[pindex], -1 - pindex)), 1)

        if debug_prev_vec is not None:
            rhd = vsa.hsim(debug_prev_vec, reverse_vec)
        return reverse_vec, pindex - 1

    def forward_unbind(self, invec):
        try:
            pindex = 0 - self.get_current_permutation(invec) - 1
            nextvec = np.roll(vsa.unbind(invec, np.roll(self.roles.permVecs[0 - pindex], pindex)), -1)
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

        try:
            # pvec = np.roll(NewChunkPvecs.permVecs[0 - ppindex], ppindex)
            nextvec2, _ = self.forward_unbind(nextvec)
        except IndexError:
            log.error("{:04d}: {} - Could not resolve Tvec to calc next vec".format(self.chunk_id, self.aname))
            # We return 0 instead of hd_match to indicate that the match was unrelaible
            # and should be discounted as a valid match when check_for_activation() returns
            return 0, invec, -1, invec

        return hd_match, nextvec, -1 - pindex, nextvec2

    @staticmethod
    def get_numeric(datavec, role_vec, stop_at):

        i = 0
        for i in range(stop_at):
            if vsa.hsim(np.roll(role_vec, i), datavec) >= CSPvec.trace_threshold:
                break

        return i

    def get_next_vector(self):
        pvec = np.roll(CSPvec[0][self.next_vec + 1], -1 * self.next_vec)
        nextvec = np.roll(vsa.bind(pvec, self.commandvec), -1)
        return nextvec

    def check_for_stopvec(self, invec):
        return vsa.hsim(self.stopvec, invec) > self.match_theshold(invec)

    def check_for_start_tag_vec(self, invec):
        return vsa.hsim(self.roles.tvec_tag, invec) > self.match_theshold(invec)

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

