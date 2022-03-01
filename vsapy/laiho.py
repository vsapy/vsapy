import vsapy
from .vsatype import *


class Laiho(VsaBase):
    vsatype = VsaType.Laiho

    @classmethod
    def default_numpy_type(cls):
        """
        :return: the default numpy datatype for this type of VSA.
        """
        return 'int16'

    @classmethod
    def unpackbits(cls, v):
        # Init sparse bound vector (s_bound) with zeros
        s_bound = np.zeros((v.slots, v.bits_per_slot))  # Create a slotted vector with

        slot_index = -1
        for bit_index in v:
            slot_index += 1
            s_bound[slot_index][bit_index] = 1

        return s_bound

    @classmethod
    def packbits(cls, v):
        # Laiho vecs are manipulated in packed format except for bundling.
        return v

    @classmethod
    def randvec(cls, dims, word_size=16,  vsa_type=VsaType.Laiho, bits_per_slot=None):
        """
        :param bits_per_slot:
        :param dims: integer or tuple, specifies shape of required array, last element is no bits per vector.
        :param word_size: numpy's word size parameter, e.g. for BSCs wordsize=8 becomes 'uint8'.
        :param vsa_type: type of VSA subclass to create from VsaType class.
        :return: a matrix of vectors of shape 'dims'.
        """
        return VsaBase(np.random.randint(0, bits_per_slot, size=dims, dtype='int' + str(word_size)),
                       vsa_type=vsa_type,
                       bits_per_slot=bits_per_slot)

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
        assert a.bits_per_slot == b.bits_per_slot, "Bits per slot mismatch"
        assert a.slots == b.slots, "not of slots mismatch"
        return (a + b) % a.bits_per_slot

    @classmethod
    def unbind(cls, a, b):
        """
        Non-Comutative unbinding operator. Decouples a from b and vice-versa. The result
        :param a: is the bundled vector
        :param b: is the role or filler vector
        :return: reverses a bind operation. If z = bind(x, y) then x = unbind(z, y) and y = unbind(z, x).
                 The return is orthogonal to x nd y if x and y have not been previously associated with bind(x, y).
        """

        # raise NotImplementedError('Laiho unbunding is NonCommutative, use self.unbind()')
        # return
        assert a.vsa_type == b.vsa_type, "Mismatch vsa_types."
        assert a.bits_per_slot == b.bits_per_slot, "Bits per slot mismatch."
        assert a.slots == b.slots, "Number of slots mismatch."

        return (a - b) % a.bits_per_slot

    # def unbind(self, b):
    #     """
    #     Non-Comutative unbinding operator. Decouples a from b and vice-versa. The result
    #
    #     :param b:
    #     :return:
    #     """
    #     a = self
    #     assert a.vsa_type == b.vsa_type, "Mismatch vsa_types"
    #     assert a.bits_per_slot == b.bits_per_slot, "Bits per slot mismatch"
    #     assert a.slots == b.slots, "not of slots mismatch"
    #     return (a - b) % a.bits_per_slot


    @classmethod
    def sum(cls, vlist, *args, **kwargs):
        """
        Maintains vsa_type custom attribute when perfoming numpy.sum()
        Todo: there is probably a better way than this.
        """
        unpacked_list = [v.unpackbits(v).flatten() for v in vlist]
        sumv = np.sum(unpacked_list, axis=0)
        return VsaBase(sumv.reshape(vlist[0].slots, vlist[0].bits_per_slot),
                       vsa_type=VsaType.Laiho, bits_per_slot=vlist[0].bits_per_slot)

    @classmethod
    def normalize(cls, sv, seqlength=None, rv=None):
        """
        Normalize the VSA vector
        :param sv: input VSA vector
        :param seqlength: Optional, the number of vectors created by summing a sequence of vectors.
                          This is required for BSCs, if not set, seqlength=None will raise and error .
        :param rv: Optional random vector, used for splitting ties on binary and ternary VSA vectors.
        :return: new VSA vector
        """

        return VsaBase(np.array([np.argmax(sv[s]) for s in range(0, sv.shape[0])]),
                       vsa_type=VsaType.Laiho, bits_per_slot=sv.shape[1])

    @classmethod
    def hdist(cls, a, b):
        """
        :param a: vsa vector
        :param b: vsa vector
        :return: normalized hamming distance between a and b. 0.0=exact match.
        """
        raise 1.0 - cls.hsim(a, b)

    @classmethod
    def hsim(cls, a, b):
        """
        :param a: vsa vector
        :param b: vsa vector
        :return: normalized hamming distance between a and b. 1.0=exact match.
        """
        assert a.vsa_type == b.vsa_type, "Mismatch vsa_types"
        if b.vsa_type == VsaType.Laiho:
            match = np.count_nonzero(a == b)
            return match / a.slots

        raise ValueError("Mismatch vsa_types")



