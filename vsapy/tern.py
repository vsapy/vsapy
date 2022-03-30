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
    def unpackbits(cls, v):
        vbin = np.unpackbits(v).astype('int8')
        vbin[vbin == 0] = -1
        return VsaBase(vbin, vsa_type=VsaType.Tern)

    @classmethod
    def packbits(cls, v):
        v[v == -1] = 0
        return np.packbits(v)

    @classmethod
    def randvec(cls, dims, word_size=8, vsa_type=VsaType.Tern):
        """
        :param dims: integer or tuple, specifies shape of required array, last element is no bits per vector.
        :param word_size: numpy's word size parameter, e.g. for BSCs wordsize=8 becomes 'uint8'.
        :param vsa_type: type of VSA subclass to create from VsaType class.
        :return: a matrix of vectors of shape 'dims'.
        """
        v = np.random.randint(0, 2, size=dims, dtype='int' + str(word_size))
        v[v == 0] = -1
        return VsaBase(v, vsa_type=vsa_type)

    @classmethod
    def bind(cls, a, b):  # actually bind/unbind for binary and ternary vecs
        """
        Comutative binding operator
        :param a: VSA vec
        :param b: VSA vec
        :return: vector associating/coupling a to b that is dissimilar to both a and b.
                 In most cases bind(a, b) is analogues to multiplication, e.g. bind(3,4)=>12.
                 If we know one of the operands we can recover the other using unbind(a,b) e.g unbind(3,12)=>4
        """
        assert a.vsa_type == VsaType.Tern or b.vsa_type == VsaType.TernZero, "Mismatch vsa_types"
        if b.vsa_type == VsaType.Tern or b.vsa_type == VsaType.TernZero:
            return -(a * b)
        raise ValueError

    @classmethod
    def unbind(cls, a, b):
        """
        Comutative unbinding operator. Decouples a from b and vice-versa. The result
        :param a: VSA vec
        :param b: VSA vec
        :return: reverses a bind operation. If z = bind(x, y) then x = unbind(y, z) and y = unbind(x, z).
                 The return is orthogonal to x nd y if x and y have not been previously associated with bind(x, y).
        """
        return cls.bind(a, b)

    @classmethod
    def normalize(cls, sv, seqlength=None, rv=None):
        """
        Normalize the VSA vector
        :param a: input VSA vector
        :param seqlength: not used
        :param rv: Optional random vector, used for splitting ties on binary and ternary VSA vectors.
        :return: new VSA vector
        """
        v = sv.copy()  # we don't want to change sv
        if rv is None:
            rv = cls.randvec(len(v), vsa_type=sv.vsa_type)
        v[v == 0] = rv[v == 0]  # Randomise the zeros to +1 or -1
        v[v < 0] = -1.0  # all elements < 0 become -1
        v[v > 0] = 1.0  # all elements > 0 become +1
        return v

    @classmethod
    def hdist(cls, a, b):
        """
        :param a: vsa vector
        :param b: vsa vector
        :return: normalized hamming distance between a and b. 0.0=exact match.
        """
        return 1.0 - cls.hsim(a,b)

    @classmethod
    def hsim(cls, a, b):
        """
        :param a: vsa vector
        :param b: vsa vector
        :return: normalized hamming distance between a and b. 1.0=exact match.
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
                num_zeros = np.count_nonzero(x == 0)  # Count zeros in the array
                hd1 = np.count_nonzero(x == 1)  # Count 1's in the array
                hd1 = (hd1 + num_zeros * 0.5) / len(x)  # Add half of the don't care bits (zeros) to teh HDSim

                #################################################################################################
                # Note the alternate way shown below changes the TernZero HDSim scale relative to Tern and BSC
                # making the HDSim value relatively higher whereas the method above matches the relative scales.
                #    num_minus1s = np.count_nonzero(x == -1)  # Count -1's in the array
                #    num_ones = np.count_nonzero(x == 1)  # Count 1's in the array
                #    hd1 = num_ones / (num_ones+num_minus1s)  # The 'active vector length' = (num_ones + num_minus1s)
            return hd1
        raise ValueError

