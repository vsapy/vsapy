
# import math
# import numpy as np
from functools import reduce

from .vsatype import *

class HRR(VsaBase):
    vsatype = VsaType.HRR

    @classmethod
    def default_numpy_type(cls):
        """
        :return: the default numpy datatype for this type of VSA.
        """
        return 'float'

    @classmethod
    def getRandVec(cls, dims, word_size=8, vsa_type=VsaType.HRR):
        """
        :param dims: integer or tuple, specifies shape of required array, last element is no bits per vector
        :return: a single vector of 'dims' bits when shape is an int, otherwise as matrix of vectors.
        """
        return VsaBase(np.random.uniform(-1.0, 1.0, dims), vsa_type=VsaType.HRR)

    @classmethod
    def bind(cls, a, b):
        assert a.vsa_type == b.vsa_type, "Mismatch vsa_types"
        if a.vsa_type == VsaType.HRR and a.vsa_type == b.vsa_type:
            return cls.cconv(a, b)
        raise ValueError("Mismatch vsa_types")

    @classmethod
    def unbind(cls, a, b):
        assert a.vsa_type == b.vsa_type, "Mismatch vsa_types"
        if b.vsa_type == VsaType.HRR:
            return cls.ccorr(a, b)
        raise ValueError("Mismatch vsa_types")

    @classmethod
    def normalizeVector(cls, a, seqlength, Rv=None):
        assert a.vsa_type == VsaType.HRR, "Mismatch vsa_types"
        if a.vsa_type == VsaType.HRR:
            return cls.normalize(a)
        raise ValueError("Mismatch vsa_types")


    @classmethod
    def HDsim(cls, a, b):
        """
        Returns hamming similarity between v1 and v2. This is equivalent to (1-hamming_distance)
        :param a:
        :param b:
        :return:
        """
        assert a.vsa_type == a.vsa_type, "Mismatch vsa_types"
        if b.vsa_type == VsaType.HRR:
            return abs(1.0 - cls.cosine(a, b))
        raise ValueError("Mismatch vsa_types")

    @classmethod
    def cconv(cls, a, b):
        """
        Computes the circular convolution of the (real-valued) vectors a and b.
        """
        assert a.vsa_type == b.vsa_type, "Mismatch vsa_types"
        return VsaBase(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real, vsa_type=VsaType.HRR)

    @classmethod
    def ccorr(cls, a, b):
        """
        Computes the circular correlation (inverse convolution) of the real-valued
        vector a with b.
        """
        assert a.vsa_type == b.vsa_type, "Mismatch vsa_types"
        return cls.cconv(np.roll(a[::-1], 1), b)

    @classmethod
    def ordConv(cls, a, b, p1, p2):
        """
        Performs ordered (non-commutative) circular convolution on the vectors a and
        b by first permuting them according to the index vectors p1 and p2.
        """
        assert all(v == a for v in [b, p1, p1]), "Mismatch vsa_types"
        return cls.cconv(a[p1], b[p2])

    @classmethod
    def convBind(cls, p1, p2, l):
        """
        Given a list of vectors, iteratively convolves them into a single vector
        (i.e., "binds" them together).
        """
        return reduce(lambda a, b: cls.normalize(cls.ordConv(a, b, p1, p2)), l)

    @classmethod
    def addBind(cls, p1, p2, l):
        """
        Given a list of vectors, binds them together by iteratively convolving them
        with place vectors and adding them up.
        """
        return cls.normalize(reduce(lambda a, b: cls.cconv(a, p1) + cls.cconv(b, p2), l))

    @classmethod
    def getRandVec_C(cls, d):
        # return VsaBase.normalize_C(np.random.randn(d) * d ** -0.5)
        return VsaBase(np.random.uniform(-1.0, 1.0, d), VsaType.HRR)

    @classmethod
    def normalize(cls, a):
        """
        Normalize a vector to length 1.
        :param a: Vector
        :return: a / len(a)
        """
        return a / np.sum(a ** 2.0) ** 0.5
