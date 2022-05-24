from enum import IntEnum
import numpy as np
import scipy.spatial as sp


class VsaType(IntEnum):
    BSC = 1  # i.i.d Binary spatter code
    Tern = 2  # i.i.d Ternary spatter code (+1, -1)
    TernZero = 3  # Ternary spatter code but allows don't care (+1, 0, -1) where 0 means tie occured in bundling
    HRR = 4  # Holographic Reduced Representation
    Laiho = 5  # Laiho full implementation
    LaihoX = 6  # Laiho simplified - the first nonzero element is taken in bundling operations


class VsaBase(np.ndarray):
    vsatype = None

    class NoAccess(Exception): pass
    class Unknown(Exception): pass

    def __new__(cls, input_array, vsa_type, dtype=None, bits_per_slot=None):
        """
        Create instance of appropriate VsaBase subclass using vsa_type.
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
            if vsa_type == VsaType.Laiho or vsa_type == VsaType.LaihoX:
                assert bits_per_slot is not None, "you must supply value for bits_per_slot"
                obj.slots = input_array.shape[-1]
                obj.bits_per_slot = bits_per_slot

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
        # if self.vsa_type == VsaType.Laiho:
        self.slots = getattr(obj, 'slots', None)
        self.bits_per_slot = getattr(obj, 'bits_per_slot', None)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(VsaBase, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.__dict__,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.__dict__.update(state[-1])  # Update the internal dict from state
        # Call the parent's __setstate__ with the other tuple elements.
        super(VsaBase, self).__setstate__(state[0:-1])


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
    def validate_operand(cls, b):
        """
        Ensure partner operand is of compatible 'vsatype'
        :param b:
        :return:
        """
        assert b.vsa_type == cls.vsatype, "Mismatch vsa_types"
        return b.vsa_type == cls.vsatype

    @staticmethod
    def trunc_vecs_to_same_len(a, b):
        """
        Because VSA vectors are holgraphic we can compare vectors of different lengths.
        this method ensures that paramters are of the same length by truncating to the shortest of the two.
        :param a: VSA vector
        :param b: VSA vector
        :return: parms (a, b) truncated to match shortest vector.
        """
        if len(a) == len(b):
            return a, b
        elif len(a) > len(b):
            return a[:len(b)], b
        else:
            return a, b[:len(a)]

    @property
    def unpack(self):
        return np.unpackbits(self)

    @property
    def pack(self):
        return VsaBase(np.packbits(self), self.vsa_type)

    @classmethod
    def random_threshold(cls, *args, **kwargs):
        """
        Should return a normalised value of the similarity match that would be expected when comparing
        random/orthorgonal vectors.
        :rtype: float
        """
        raise NotImplementedError('Subclass must implment "random_threshold()"')


    @classmethod
    def randvec(cls, *args, **kwargs):
        """
        :param dims: integer or tuple, specifies shape of required array, last element is no bits per vector.
        :param word_size: numpy's word size parameter, e.g. for BSCs wordsize=8 becomes 'uint8'.
        :param vsa_type: type of VSA subclass to create from VsaType class.
        :return: a matrix of vectors of shape 'dims'.
        """
        raise NotImplementedError('Subclass must implment "randvec()"')

    @classmethod
    def normalize(cls, sv, *args, **kwargs):
        """
        Normalize the VSA vector
        :param a: input VSA vector
        :param seqlength: Optional, needed to normalise BSC vectors.
        :param rv: Optional random vector, used for splitting ties on binary and ternary vectors.
        :return: new VSA vector
        """
        raise NotImplementedError('Subclass must implment "normalize()"')

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
        raise NotImplementedError('Subclass must implment "bind()"')

    @classmethod
    def unbind(cls, a, b):
        """
        Comutative unbinding operator. Decouples a from b and vice-versa. The result
        :param a: VSA vec
        :param b: VSA vec
        :return: reverses a bind operation. If z = bind(x, y) then x = unbind(y, z) and y = unbind(x, z).
                 The return is orthogonal to x nd y if x and y have not been previously associated with bind(x, y).
        """
        raise NotImplementedError('Subclass must implment "unbind()"')

    @classmethod
    def hdist(cls, a, b):
        """
        :param a: vsa vector
        :param b: vsa vector
        :return: normalized hamming distance between a and b. 0.0=exact match.
        """
        raise NotImplementedError('Subclass must implment "hdist()"')

    @classmethod
    def hsim(cls, a, b):
        """
        :param a: vsa vector
        :param b: vsa vector
        :return: normalized hamming similarity between a and b. 1.0=exact match.
        """
        raise NotImplementedError('Subclass must implment "hsim()"')


    @classmethod
    def cosine(cls, a, b):
        """
        :param a: vsa vector
        :param b: vsa vector
        :return: cosine distance between a and b, 0.0=exact match.
        """
        if a.vsa_type == VsaType.Tern or a.vsa_type == VsaType.TernZero:
            assert b.vsa_type == VsaType.Tern or b.vsa_type == VsaType.TernZero, "Mismatch vsa_types"
        else:
            assert a.vsa_type == b.vsa_type, "Mismatch vsa_types"
        return sp.distance.cosine(a, b)

    @classmethod
    def cosine_sim(cls, a, b):
        """
        :param a: vsa vector
        :param b: vsa vector
        :return: cosine similarity between a and b. 1.0=exact match.
        """
        return 1.0 - cls.cosine(a, b)

    @classmethod
    def sum(cls, ndarray, *args, **kwargs):
        """
        Maintains vsa_type custom attribute when perfoming numpy.sum()
        Todo: there is probably a better way than this.
        """
        return VsaBase(np.sum(np.array(ndarray), axis=0, *args, **kwargs), vsa_type=ndarray[0].vsa_type)
