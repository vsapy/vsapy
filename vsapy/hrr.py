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
    def unpackbits(cls, v):
        # HRR vecs are not packed
        return v

    @classmethod
    def packbits(cls, v):
        # HRR vecs are not packed
        return v

    @classmethod
    def randvec(cls, dims, word_size=8, vsa_type=VsaType.HRR):
        """
        :param dims: integer or tuple, specifies shape of required array, last element is no bits per vector.
        :param word_size: numpy's word size parameter, e.g. for BSCs wordsize=8 becomes 'uint8'.
        :param vsa_type: type of VSA subclass to create from VsaType class.
        :return: a matrix of vectors of shape 'dims'.
        """
        return VsaBase(np.random.uniform(-1.0, 1.0, dims), vsa_type=VsaType.HRR)

    @classmethod
    def bind(cls, a, b):
        """
        Comutative binding operator
        :param a: VSA vec
        :param b: VSA vec
        :return: vector associating/coupling a to b that is dissimilar to both a and b.
                 In most cases bind(a, b) is analogues to multiplication, e.g. bind(3,4)=>12.
                 If we know one of the operands we can recover the other using unbind(a,b) e.g unbind(3,12)=>4
        """
        assert a.vsa_type == b.vsa_type, "Mismatch vsa_types"
        if a.vsa_type == VsaType.HRR and a.vsa_type == b.vsa_type:
            return cls.cconv(a, b)
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
        assert a.vsa_type == b.vsa_type, "Mismatch vsa_types"
        if b.vsa_type == VsaType.HRR:
            return cls.ccorr(a, b)
        raise ValueError("Mismatch vsa_types")

    @classmethod
    def normalize(cls, sv, seqlength=None, rv=None):
        """
        Normalize the VSA vector
        :param a: input VSA vector
        :param seqlength: not used
        :param rv: Optional random vector, used for splitting ties on binary and ternary VSA vectors.
        :return: new VSA vector
        """
        assert sv.vsa_type == VsaType.HRR, "Mismatch vsa_types"
        if sv.vsa_type == VsaType.HRR:
            """
            Normalize a vector to length 1.
            :param a: Vector
            :return: a / len(a)
            """
            v = np.asarray(sv)
            return VsaBase(v / np.sum(v ** 2.0) ** 0.5, vsa_type=VsaType.HRR)
        raise ValueError("Mismatch vsa_types")

    @classmethod
    def hdist(cls, a, b):
        """
        Cosine is used for hsim() for real number vectors
        :param a: vsa vector
        :param b: vsa vector
        :return: cosine similarity between a and b. 0.0=exact match.
        """
        return cls.cosine(a, b)

    @classmethod
    def hsim(cls, a, b):
        """
        Cosine is used for hsim() for real number vectors
        :param a: vsa vector
        :param b: vsa vector
        :return: cosine similarity between a and b. 1.0=exact match.
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
