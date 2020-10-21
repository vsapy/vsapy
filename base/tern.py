from functools import reduce
import numpy as np

from .vsatype import *

class Tern(VsaBase):
    vsatype = VsaType.Tern

    @classmethod
    def default_numpy_type(cls):
        """
        :return: the default numpy datatype for this type of VSA.
        """
        return 'int8'

    @classmethod
    def getRandVec(cls, dims, word_size=8, vsa_type=VsaType.Tern):
        """
        :param dims: integer or tuple, specifies shape of required array, last element is no bits per vector
        :return: a single vector of 'dims' bits when shape is an int, otherwise as matrix of vectors.
        """
        v = np.random.randint(0, 2, size=dims, dtype='int' + str(word_size))
        v[v == 0] = -1
        return VsaBase(v, vsa_type=vsa_type)

    @classmethod
    def bind(cls, a, b):  # actually bind/unbind for binary and ternary vecs
        assert a.vsa_type == VsaType.Tern or b.vsa_type == VsaType.TernZero, "Mismatch vsa_types"
        if b.vsa_type == VsaType.Tern or b.vsa_type == VsaType.TernZero:
            return -(a * b)
        raise ValueError

    @classmethod
    def unbind(cls, a, b):
        return cls.bind(a, b)

    @classmethod
    def normalizeVector(cls, sv, seqlength, Rv=None):
        v = sv.copy()  # we don't want to change sv
        if Rv is None:
            Rv = cls.getRandVec(len(v), vsa_type=sv.vsa_type)
        v[v == 0] = Rv[v == 0]  # Randomise the zeros to +1 or -1
        v[v < 0] = -1.0  # all elements < 0 become -1
        v[v > 0] = 1.0  # all elements > 0 become +1
        return v


    @classmethod
    def HDsim(cls, a, b):
        """
        Returns hamming similarity between v1 and v2. This is equivalent to (1-hamming_distance)
        :param a:
        :param b:
        :return:
        """
        assert b.vsa_type == VsaType.Tern or b.vsa_type == VsaType.TernZero, "Mismatch vsa_types"
        if b.vsa_type == VsaType.Tern or b.vsa_type == VsaType.TernZero:
            x = cls.bind(a, b) * -1

            if np.count_nonzero(x) == len(x):
                num_ones = (x == 1)  # Gets an array of the +1's only (will be shorter than the full length)
                return np.count_nonzero(num_ones) / len(x)
            else:
                # For TernZero class, if there are zeros allowed in the array we can simulate the HDSim like below.
                # Doing it this way means we keep the HDSim scale relative to BSC and Tern
                # num_zeros = np.count_nonzero(x == 0)  # Count zeros in the array
                # hd1 = np.count_nonzero(x == 1)  # Count 1's in the array
                # hd1 = (hd1 + num_zeros * 0.5) / len(x)  # Add half of the don't care bits (zeros) to teh HDSim

                # An alternate maybe to do hamming distance on the active bits
                num_minus1s = np.count_nonzero(x == -1)  # Count -1's in the array
                num_ones = np.count_nonzero(x == 1)  # Count 1's in the array
                hd1 = num_ones / (num_ones+num_minus1s)  # The active vector length is equal to num_ones + num_minus1s
            return hd1
        raise ValueError

