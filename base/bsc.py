from functools import reduce
import numpy as np
from scipy import special as scm
import scipy.spatial as sp

from .vsatype import *


class BSC(VsaBase):
    vsatype = VsaType.BSC

    @classmethod
    def default_numpy_type(cls):
        """
        :return: the default numpy datatype for this type of VSA.
        """
        return 'uint8'

    @classmethod
    def getRandVec(cls, dims, word_size=8, vsa_type=VsaType.BSC):
        """
        :param dims: integer or tuple, specifies shape of required array, last element is no bits per vector
        :return: a single vector of 'dims' bits when shape is an int, otherwise as matrix of vectors.
        """
        return VsaBase(np.random.randint(0, 2, size=dims, dtype='uint' + str(word_size)), vsa_type=vsa_type)

    @classmethod
    def bind(cls, a, b):  # actually bind/unbind for binary and ternary vecs
        assert a.vsa_type == b.vsa_type, "Mismatch vsa_types"
        if a.vsa_type == VsaType.BSC and a.vsa_type == b.vsa_type:
            return np.logical_xor(a, b) * 1
        raise ValueError("Mismatch vsa_types")

    @classmethod
    def unbind(cls, a, b):
        return cls.bind(a, b)

    @classmethod
    def normalizeVector(cls, sv, seqlength, Rv=None):
        v = sv.copy()  # we don't want to change sv
        if seqlength % 2 == 0:
            if Rv is None:
                v = v + cls.getRandVec(len(v))
            else:
                v = v + Rv
            seqlength += 1

        v[v < float(seqlength / 2.0)] = 0  # using cast because of running in python 2.7
        v[v > float(seqlength / 2.0)] = 1

        return v

    @classmethod
    def HDsim(cls, a, b):
        """
        Returns hamming similarity between v1 and v2. This is equivalent to (1-hamming_distance)
        :param a:
        :param b:
        :return:
        """
        assert a.vsa_type == b.vsa_type, "Mismatch vsa_types"
        if b.vsa_type == VsaType.BSC:
            return 1.0 - float(np.count_nonzero(np.logical_xor(a, b))) / len(a)
        raise ValueError("Mismatch vsa_types")


