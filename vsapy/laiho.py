import math
from scipy import stats
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
    def is_laiho_type(cls, vsa_type):
        if isinstance(vsa_type, vsapy.BagVec):
            return isinstance(vsa_type.myvec, Laiho)
        elif isinstance(vsa_type, Laiho):
            return True
        if vsa_type == VsaType.Laiho or vsa_type == VsaType.LaihoX:
            return True
        return False

    @classmethod
    def expandbits(cls, v):
        # Init sparse bound vector (s_bound) with zeros
        s_bound = np.zeros((v.slots, v.bits_per_slot))  # Create a slotted vector with

        slot_index = -1
        for bit_index in v:
            slot_index += 1
            s_bound[slot_index][bit_index] = 1

        return s_bound

    @classmethod
    def unpackbits(cls, v):
        # Laiho vecs are manipulated in packed format.
        return v

    @classmethod
    def packbits(cls, v):
        # Laiho vecs are manipulated in packed format.
        return v

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
            if isinstance(args[0], Laiho):
                slots = len(args[0])
                bits_per_slot = args[0].bits_per_slot
            else:
                raise ValueError("Expected a Laiho/LaihoX vector.")
        else:
            slots = kwargs.get('slots', -1)
            if slots < 1:
                raise ValueError("You must supply a sample vector, or set optional parameter 'slots'")
            bits_per_slot = kwargs.get('bits_per_slot', -1)
            if bits_per_slot < 1:
                raise ValueError("You must supply a sample vector, or set optional parameter 'bits_per_slot'")

        p = 1 / bits_per_slot  # Probability of a match between random vectors
        var_rv = slots * (p * (1 - p))  # Varience (un-normalised)
        std_rv = math.sqrt(var_rv)  # Stdev (un-normalised)
        hdrv = slots / bits_per_slot + stdev_count * std_rv  # Un-normalised hsim of two randomvectors adjusted by 'n' stdevs
        return hdrv / slots

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
    def sum2(cls, vlist, *args, **kwargs):
        """
        Maintains vsa_type custom attribute when perfoming numpy.sum()
        Todo: there is probably a better way than this.
        """
        # Super slow method compared to sum() but was used as a sanity checker.
        unpacked_list = [v.expandbits(v).flatten() for v in vlist]
        orvec = np.sum(unpacked_list, axis=0).reshape(vlist[0].slots, vlist[0].bits_per_slot)
        outvec = np.ma.masked_array(orvec, (orvec == 0)).argmax(axis=1)

        return VsaBase(np.array(outvec), vsa_type=VsaType.Laiho, bits_per_slot=vlist[0].bits_per_slot)

    @classmethod
    def sum1(cls, vlist, *args, **kwargs):
        """
        Maintains vsa_type custom attribute when perfoming numpy.sum()
        Todo: there is probably a better way than this.
        """

        # This method is 2X slower than helpers.py mode() !
        return VsaBase(stats.mode(vlist)[0][0], vsa_type=VsaType.Laiho, bits_per_slot=vlist[0].bits_per_slot)

    @classmethod
    def sum(cls, vlist, *args, **kwargs):
        """
        Maintains vsa_type custom attribute when perfoming numpy.sum()
        Todo: there is probably a better way than this.
        """
        # helpers.py mode() is 2x faster than scipy.stats.mode() !
        return VsaBase(mode(vlist, axis=0)[0], vsa_type=VsaType.Laiho, bits_per_slot=vlist[0].bits_per_slot)

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
        if isinstance(b, Laiho):
            match = np.count_nonzero(a == b)
            return match / a.slots

        raise ValueError("Mismatch vsa_types")

    @staticmethod
    def slots_from_bsc_vec(bsc_dim, bits_per_slot):
        return int(bsc_dim / math.log(bits_per_slot, 2))

    @staticmethod
    def bits_from_bsc_vec(bsc_dim, slots):
        return int(2 ** (bsc_dim/slots))

    @staticmethod
    def bsc_dim_from_laiho(slots, bits_per_slot):
        return int(slots * math.log(bits_per_slot, 2))


def mode(ndarray, axis=0):
    # Check inputs
    ndarray = np.asarray(ndarray)
    ndim = ndarray.ndim
    if ndarray.size == 1:
        return (ndarray[0], 1)
    elif ndarray.size == 0:
        raise Exception('Cannot compute mode on empty array')
    try:
        axis = range(ndarray.ndim)[axis]
    except:
        raise Exception('Axis "{}" incompatible with the {}-dimension array'.format(axis, ndim))

    # If array is 1-D and numpy version is > 1.9 np.unique will suffice
    if all([ndim == 1,
            int(np.__version__.split('.')[0]) >= 1,
            int(np.__version__.split('.')[1]) >= 9]):
        modals, counts = np.unique(ndarray, return_counts=True)
        index = np.argmax(counts)
        return modals[index], counts[index]

    # Sort array
    sort = np.sort(ndarray, axis=axis)
    # Create array to transpose along the axis and get padding shape
    transpose = np.roll(np.arange(ndim)[::-1], axis)
    shape = list(sort.shape)
    shape[axis] = 1
    # Create a boolean array along strides of unique values
    strides = np.concatenate([np.zeros(shape=shape, dtype='bool'),
                                 np.diff(sort, axis=axis) == 0,
                                 np.zeros(shape=shape, dtype='bool')],
                                axis=axis).transpose(transpose).ravel()
    # Count the stride lengths
    counts = np.cumsum(strides)
    counts[~strides] = np.concatenate([[0], np.diff(counts[~strides])])
    counts[strides] = 0
    # Get shape of padded counts and slice to return to the original shape
    shape = np.array(sort.shape)
    shape[axis] += 1
    shape = shape[transpose]
    slices = [slice(None)] * ndim
    slices[axis] = slice(1, None)
    # Reshape and compute final counts
    counts = counts.reshape(shape).transpose(transpose)[tuple(slices)] + 1

    # Find maximum counts and return modals/counts
    slices = [slice(None, i) for i in sort.shape]
    del slices[axis]
    index = np.ogrid[slices]
    index.insert(axis, np.argmax(counts, axis=axis))
    return sort[tuple(index)], counts[tuple(index)]