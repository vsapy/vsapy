from enum import IntEnum
import numpy as np
import scipy.spatial as sp


class VsaType(IntEnum):
    BSC = 1  # i.i.d Binary spatter code
    Tern = 2  # i.i.d Ternary spatter code (+1, -1)
    TernZero = 3  # Ternary spatter code but allows don't care (+1, 0, -1) where 0 means tie occured in bundling
    HRR = 4  # Holographic Reduced Representation


class VsaBase(np.ndarray):
    vsatype = None

    class NoAccess(Exception): pass
    class Unknown(Exception): pass

    def __new__(cls, input_array, vsa_type, dtype=None):
        """
        Create instance of appropriate subclass using path prefix.

        :param input_array:
        :param vsa_type: subclass of VsaBase to be created from VsaType class
        :param dtype: numpy dtype
        """
        subclass = cls.get_subclass(vsa_type)
        if subclass:
            # Using "object" base class method avoids recursion here.
            obj = np.asarray(input_array).view(subclass)
            # add the new attribute to the created instance
            obj.vsa_type = vsa_type
            # Finally, we must return the newly created object, cast to the required numpy dtype:
            if dtype:
                return obj.astype(dtype)
            else:
                return obj.astype(obj.default_numpy_type())

        else:  # No subclass with matching prefix found (& no default defined)
            raise VsaBase.Unknown(
                'ERROR: class "VsaType.{}" is not defined.'.format(vsa_type.name))

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.vsa_type = getattr(obj, 'vsa_type', None)


    @classmethod
    def default_numpy_type(cls):
        """

        :return: Should return the default numpy datatype for this type of VSA, e.g., for Vsa.BSC return 'uint8'
        """
        raise NotImplementedError('Subclass must implment property "default_numpy_type()"')

    @classmethod
    def _get_all_subclasses(cls):
        """ Recursive generator of all class' subclasses. """
        for subclass in cls.__subclasses__():
            yield subclass
            for subclass in subclass._get_all_subclasses():
                yield subclass

    @classmethod
    def get_subclass(cls, vsa_type):
        for subclass in cls._get_all_subclasses():
            if subclass.vsatype == vsa_type:
                return subclass
        return None

    @classmethod
    def get_all_converters(cls):
        converters = {}
        for subclass in cls._get_all_subclasses():
            converters[subclass.vsatype] = subclass.get_converters()

        return converters

    @classmethod
    def get_converter(cls, vsa_type):
        raise NotImplementedError('Subclass must implment "validate_operand()"')

    @classmethod
    def validate_operand(cls, v2):
        assert v2.vsa_type == cls.vsatype, "Mismatch vsa_types"
        return (v2.vsa_type == cls.vsatype)

    @staticmethod
    def trunc_vecs_to_same_len(a, b):
        if len(a) == len(b):
            return a, b
        elif len(a) > len(b):
            return a[:len(b)], b
        else:
            return a, b[:len(a)]

    @classmethod
    def getRandVec(cls, dims, word_size=8, vsa_type=VsaType.BSC):
        """

        :param dims: integer or tuple, specifies shape of required array, last element is no bits per vector
        :param word_size: numpy's word size parameter, e.g. for BSCs wordsize=8 becomes 'uint8'
        :param vsa_type: type of VSA subclass to create from VsaType class
        :return: a single vector of 'dims' bits when shape is an int, otherwise as matrix of vectors.
        """
        """
        :param dims: integer or tuple, specifies shape of required array, last element is no bits per vector
        :return: a single vector of 'dims' bits when shape is an int, otherwise as matrix of vectors.
        """

        raise NotImplementedError('Subclass must implment "validate_operand()"')

    @classmethod
    def normalizeVector(cls, sv, seqlength, Rv=None):
        raise NotImplementedError('Subclass must implment "validate_operand()"')

    @classmethod
    def bind(cls, a, b):
        raise NotImplementedError('Subclass must implment "validate_operand()"')

    @classmethod
    def unbind(cls, a, b):
        raise NotImplementedError('Subclass must implment "validate_operand()"')

    @classmethod
    def cosine(cls, a, b):
        if a.vsa_type == VsaType.Tern or a.vsa_type == VsaType.TernZero:
            assert b.vsa_type == VsaType.Tern or b.vsa_type == VsaType.TernZero, "Mismatch vsa_types"
        else:
            assert a.vsa_type == b.vsa_type, "Mismatch vsa_types"
        return sp.distance.cosine(a, b)

    @classmethod
    def HDsim(cls, a, b):
        """
        Returns hamming similarity between v1 and v2. This is equivalent to (1-hamming_distance)
        :param a:
        :param b:
        :return:
        """
        raise NotImplementedError('Subclass must implment "validate_operand()"')

