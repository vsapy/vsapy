import math

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
    def unpackbits(cls, v):
        return VsaBase(np.unpackbits(v), vsa_type=VsaType.BSC)

    @classmethod
    def packbits(cls, v):
        return np.packbits(v)

    @classmethod
    def random_threshold(cls, *args, stdev_count=4.4, **kwargs):
        """
        Should return a normalised value of the similarity match that would be expected when comparing
        random/orthorgonal vectors.

        :param args: A sample vector the dimensions of which should be used to calculate threshold.
        :param stdev_count:
        :param kwargs: slots=<int>, bits_per_slot=<int>
        :return: normalised threshold value
        :rtype: float
        """
        if len(args) > 0:
            if isinstance(args[0], BSC):
                D = len(args[0])
            elif isinstance(args[0], int):
                D = args[0]
            else:
                raise ValueError("Expected a BSC vector or an int specifying vector dimension.")
        else:
            raise ValueError("You must supply a sample vector, or a value for dimensionality.")

        p = 0.5  # Probability of a match between random vectors
        var_rv = D * (p * (1 - p))  # Varience (un-normalised)
        std_rv = math.sqrt(var_rv)  # Stdev (un-normalised)
        hdrv = D * 0.5 + stdev_count * std_rv  # Un-normalised hsim of two randomvectors adjusted by 'n' stdevs
        return hdrv / D

    @classmethod
    def randvec(cls, dims, word_size=8, vsa_type=VsaType.BSC):
        """
        :param dims: integer or tuple, specifies shape of required array, last element is no bits per vector.
        :param word_size: numpy's word size parameter, e.g. for BSCs wordsize=8 becomes 'uint8'.
        :param vsa_type: type of VSA subclass to create from VsaType class.
        :return: a matrix of vectors of shape 'dims'.
        """
        return VsaBase(np.random.randint(0, 2, size=dims, dtype='uint' + str(word_size)), vsa_type=vsa_type)

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
        assert a.vsa_type == b.vsa_type, "Mismatch vsa_types"
        if a.vsa_type == VsaType.BSC and a.vsa_type == b.vsa_type:
            return np.logical_xor(a, b) * 1
        raise ValueError("Mismatch vsa_types")

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
    def normalize(cls, sv, seqlength, *args,  rv=None, **kwargs):
        """
        Normalize the VSA vector
        :param sv: input VSA vector
        :param seqlength: Optional, the number of vectors created by summing a sequence of vectors.
                          This is required for BSCs, if not set, seqlength=None will raise and error .
        :param rv: Optional random vector, used for splitting ties on binary and ternary VSA vectors.
        :return: new VSA vector
        """
        if seqlength == 1:
            return sv

        v = sv.copy()  # we don't want to change sv
        assert seqlength is not None, "You must specify the sequence length"
        if seqlength % 2 == 0:  # This will throw and error if seqlength=None. Forces parameter to be passed valid value
            if rv is None:
                v = v + cls.randvec(len(v))
            else:
                v = v + rv
            seqlength += 1

        v[v < float(seqlength / 2.0)] = 0  # using cast because of running in python 2.7
        v[v > float(seqlength / 2.0)] = 1

        return v

    @classmethod
    def hdist(cls, a, b):
        """
        :param a: vsa vector
        :param b: vsa vector
        :return: normalized hamming distance between a and b. 0.0=exact match.
        """
        assert a.vsa_type == b.vsa_type, "Mismatch vsa_types"
        if b.vsa_type == VsaType.BSC:
            return float(np.count_nonzero(np.logical_xor(a, b))) / len(a)
        raise ValueError("Mismatch vsa_types")

    @classmethod
    def hsim(cls, a, b):
        """
        :param a: vsa vector
        :param b: vsa vector
        :return: normalized hamming distance between a and b. 1.0=exact match.
        """
        assert a.vsa_type == b.vsa_type, "Mismatch vsa_types"
        if b.vsa_type == VsaType.BSC:
            return 1.0 - cls.hdist(a, b)
        raise ValueError("Mismatch vsa_types")



