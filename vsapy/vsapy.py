import math
import random
import vsapy as vsa
from vsapy import *
from .vsatype import *
from vsapy.role_vectors import TimeStamp
from .bsc_stats import subvec_mean as bsc_mean
from .sparse_stats import subvec_mean as snn_mean


def random_threshold(*args, **kwargs):
    return args[0].random_threshold(*args, **kwargs)


def subvec_mean_hd(num_vecs):
    return 1.0 - subvec_mean(num_vecs)


def subvec_mean(sub_vecs, vsa_type=None, bits_per_slot=None):
    """

    :param sub_vecs: This is the number of vectors to be added via majority_sum or a bundled vector (subclass of BagVec)
    containing a vector count.
    :type sub_vecs:
    :param vsa_type: when passing a vector count, specifies the type of vecrtors being bundled
    :type vsa_type:
    :return: expected mean value of any subvector in a bundle of num_vecs=sub_vecs.
    :rtype:
    """

    if not isinstance(sub_vecs, vsa.BagVec):
        num_vecs = sub_vecs
    else:
        num_vecs = sub_vecs.vec_cnt
        if vsa_type is not None and vsa_type != sub_vecs.vsa_type:
            raise ValueError("sub_vecs should be in integer when passing a vsa_type.")
        vsa_type = sub_vecs.vsa_type
        if vsa.Laiho.is_laiho_type(sub_vecs):
            bits_per_slot = sub_vecs.myvec.bits_per_slot

    if vsa_type:
        if vsa_type == VsaType.HRR:
            raise NotImplementedError(f'subvec_mean not implemented for type{VsaType.HRR}.')
        elif vsa.Laiho.is_laiho_type(vsa_type):
            return snn_mean(num_vecs, bits_per_slot, 1)
        else:
            return bsc_mean(num_vecs)

    raise ValueError("vsa_type must be specifed.")


class NumberLine(object):
    def __init__(self, min_number, max_number, vec_dim, vsa_kwargs, quantise_interval=0, creation_data_time_stamp=None):
        """
        The NumberLine is only implemented for BSC and Tern/TernZero

        :param min_number: Min number of the range the number line will represent
        :param max_number: Max number of the range the number line will represent
        :param vec_dim: vector dimension
        :param vsa_kwargs: to pass in VSA type info.
        :param quantise_interval: sets the number of bits between integer steps.
                                  Default=0, sets the max bits between steps.
        :param creation_data_time_stamp:
        """

        if vsa_kwargs['vsa_type'] not in [VsaType.BSC, VsaType.Tern, VsaType.TernZero]:
            raise NotImplementedError(f'NumberLine is not implemented for type: {VsaType.HRR}.')

        if creation_data_time_stamp is None:
            self.creation_data_time_stamp = TimeStamp.get_creation_data_time_stamp()
        else:
            self.creation_data_time_stamp = creation_data_time_stamp

        self.min_num = min_number
        self.max_num = max_number
        self.range = max_number - min_number
        if quantise_interval > vec_dim // 2:
            raise ValueError("parameter 'quantise_interval' must be <= vec_dim // 2")
        self.Q = vec_dim // 2 if quantise_interval == 0 else quantise_interval
        self.zero_vec = vsa.randvec(vec_dim, **vsa_kwargs)
        zvec = vsa.to_vsa_type(self.zero_vec, new_vsa_type=VsaType.BSC)  # linear_sequence_gen() uses BSC vecs.
        if self.Q:
            self.number_vecs = vsa.to_vsa_type(
                VsaBase(NumberLine.linear_sequence_gen(self.Q, zvec), vsa_type=VsaType.BSC),
                new_vsa_type=vsa_kwargs['vsa_type']
            )
        else:
            self.number_vecs = vsa.to_vsa_type(
                VsaBase(NumberLine.linear_sequence_gen(max_number-min_number, zvec), **vsa_kwargs),
                new_vsa_type=vsa_kwargs['vsa_type']
            )
        self.make_orthogonal = vsa.randvec(vec_dim, **vsa_kwargs)

    def number_to_vec(self, num):
        # If number is out of range, report this
        # The alternative is to quantise

        if num < self.min_num:
            assert False, "Number is out of range."
        if num > self.max_num:
            assert False, "Number is out of range."

        if self.Q:
            # Normalise the input number to fit within the number line
            norm_n = int((num - self.min_num) / self.range * self.Q)
            return vsa.bind(self.number_vecs[norm_n], self.make_orthogonal)
        else:
            return vsa.bind(self.number_vecs[num-self.min_num], self.make_orthogonal)

    def number_from_vec(self, number_vec):
        raw_vec = vsa.bind(self.make_orthogonal, number_vec)
        r = np.apply_along_axis(np.logical_xor, 1, self.number_vecs, raw_vec)
        res = np.count_nonzero(r.astype(int), axis=1)

        # Calculate the losses per trail. each trail is a test of random_vec_HD vs myHD
        winner_indx = np.argmin(res)
        if self.Q:
            return int(winner_indx*self.range/self.Q + self.min_num)
        else:
            return winner_indx+self.min_num

    @staticmethod
    def linear_sequence_gen(max_number, start_vec):
        """

        Parameters
        ----------
        max_number : terminal number
        start_vec : base starting vector

        Returns
        -------
        return list of vectors having equal hamming distances between each vector
        """

        def gen_next_vec(in_vec, change_map, flipbits):
            out_vec = in_vec.copy()
            imap = np.argwhere(change_map == 0).ravel()
            nobits = len(imap)
            if nobits == 0:
                nobits = 1
                # return out_vec, change_map
            flip_percent = float(flipbits) / float(nobits)
            for i in imap:
                if random.random() < flip_percent:
                    out_vec[i] = not out_vec[i]
                    change_map[i] = 1

            return out_vec, change_map

        vec_len = len(start_vec) // 2  # We use half the range because orthogonal vecs have HD=0.5
        number_line = [start_vec]
        change_map = np.zeros(len(number_line[0]), dtype=int)
        if max_number > vec_len:
            # Here we need to quantise the interval because we can only have vec_len steps
            assert False, "Not implemented yet"
            steps = vec_len
        else:
            flip = vec_len // max_number
        print('No bits to flip', flip)
        for i in range(1, max_number+1):
            z, change_map = gen_next_vec(number_line[i - 1], change_map, flip)
            number_line.append(z)

        return number_line


class Real2Binary(object):
    def __init__(self, rdim, bdim, seed):
        """
        Note when converting a 'bank' / database of realnumber vectors the same seed MUST be used
        in order to ensure that the semantic vector space distances are maintatined.
        Obviously a single run will maintain this since we generate the mapper on initialisation.

        :param rdim: Dimension of the real number vec being converted
        :param bdim: Dimension of the equivalent binary vector we want to create
        :param seed: for repeatability if needed during research and debug etc
        """
        if seed:
            np.random.seed(seed)
        self.mapper = np.random.randint(0, 2, size=(bdim, rdim), dtype='uint8')

    def to_bin(self, v):
        """
        To create the binary vector multiply the mapper matrix by the real number vector.
        The random bit patterns in self.mapper * v produces a (bdim * rdim) real number matrix
        We then sum along axis=1 which gives us a 'bdim' realnumber vector.
        This is then thresholded to produce a binary bit pattern that maintains the distances in the vector space.
        The binary vector produced has an, approximately, equal number of 1's and 0's maininting thus maintaining the
        i.i.d random distribution of bits within the vector.

        example
                    2d real number vec R = [0.3, -0.7]
                    5D binary mapper   B = [[1, 0],
                                           [1, 1],
                                           [0, 0],
                                           [1, 0],
                                           [1, 1]]

                                  R * B = [[0.3, 0],
                                          [0.3, -0.7],
                                          [0.0,  0.0],  Sum along axis=1 ==> rr = [0.3, -0.4, 0.0, 0.3, -0.4]
                                          [0.3,  0.0],
                                          [0.3, -0.71]

                    We the perform thresholding and normalisation on 'rr' to convert this to a binary presentation ZZ,
                    note,

                                       ZZ = [1, 0, 1, 1, 0]

        :param v: real number vector to convert.
        :return: Binary vector representation of v having an i.i.d, approx equal number of 1's and 0's.
        """
        Exp_V = 0.5 * np.sum(v)
        Var_V = math.sqrt(0.25 * np.sum(v * v))
        ZZ = (np.sum(self.mapper * v, axis=1) - Exp_V) / Var_V  # Sum and threshold.
        # Normalise this to binary
        ZZ[ZZ >= 0.0] = 1
        ZZ[ZZ < 0.0] = 0

        return ZZ.astype('uint8')


def to_vsa_type(sv, new_vsa_type):
    """

    :param sv: 1D or 2D array of vectors
    :param new_vsa_type: Type we want the vector to become
    :return:
    """

    if sv.vsatype == new_vsa_type:
        return sv

    v = sv.copy()  # Get a copy so we do not change the source
    if sv.vsa_type == VsaType.TernZero:
        # We need to flip any zeros to a random 1 or -1
        v.vsa_type = VsaType.Tern
        v = v.reset_zeros_normalize(v)  # By Normalising as a VsaType.TERNARY we randomly flip 0's to 1 or -1
        if new_vsa_type == VsaType.Tern:
            return VsaBase(v, vsa_type=new_vsa_type)
        elif new_vsa_type == VsaType.BSC:
            v[v == -1] = 0
            v.vsa_type = VsaType.BSC  # set new vsa_type
            return VsaBase(v, vsa_type=new_vsa_type)
        else:
            raise ValueError

    if sv.vsa_type == VsaType.Tern:
        if new_vsa_type == VsaType.TernZero:
            # At VsaTernary does not have any zeros so we can hust flip the type
            return VsaBase(v, new_vsa_type)
        elif new_vsa_type == VsaType.BSC:
            v[v == -1] = 0
            v = v.astype('uint8')
            return VsaBase(v, vsa_type=new_vsa_type)
        else:
            raise ValueError

    if sv.vsa_type == VsaType.BSC:
        if new_vsa_type == VsaType.Tern or new_vsa_type == VsaType.TernZero:
            v = v.astype('int8')
            v[v == 0] = -1
            return VsaBase(v, vsa_type=new_vsa_type)

    raise ValueError(f"cannot convert from {str(sv.vsa_type)} to {str(new_vsa_type)}")


def randvec(dims, *args, **kwargs):
    """
    :param dims: integer or tuple, specifies shape of required array, last element is no bits per vector.
    :param word_size: numpy's word size parameter, e.g. for BSCs wordsize=8 becomes 'uint8'.
    :param vsa_type: type of VSA subclass to create from VsaType class.
    :return: a matrix of vectors of shape 'dims'.
    """
    vsa_type = kwargs.get("vsa_type", VsaType.BSC)
    subclass = VsaBase.get_subclass(vsa_type=vsa_type)
    if subclass:
        return subclass.randvec(dims, *args, **kwargs)
    else:
        raise ValueError


def normalize(a, *args, **kwargs):
    """
    Normalize the VSA vector
    :param a: input VSA vector
    :param seqlength: Optional, for BSC vectors must be set to a valid.
    :param rv: Optional random vector, used for splitting ties on binary and ternary VSA vectors.
    :return: new VSA vector
    """
    return a.normalize(a, *args, **kwargs)


def bind(a, b):
    """
    Comutative binding operator
    :param a: VSA vec
    :param b: VSA vec
    :return: vector associating/coupling a to b that is dissimilar to both a and b.
             In most cases bind(a, b) is analogues to multiplication, e.g. bind(3,4)=>12.
             If we know one of the operands we can recover the other using unbind(a,b) e.g unbind(3,12)=>4
    """
    if a.validate_operand(b):
        a1, b1 = VsaBase.trunc_vecs_to_same_len(a, b)
        return a.bind(a1, b1)


def unbind(a, b):  # actually bind/unbind for binary and ternary vecs
    """
    Comutative unbinding operator. Decouples a from b and vice-versa. The result
    :param a: VSA vec
    :param b: VSA vec
    :return: reverses a bind operation. If z = bind(x, y) then x = unbind(y, z) and y = unbind(x, z).
             The return is orthogonal to x nd y if x and y have not been previously associated with bind(x, y).
    """
    if a.validate_operand(b):
        a1, b1 = VsaBase.trunc_vecs_to_same_len(a, b)
        return a.unbind(a1, b1)


def cosine(a, b):
    """
    :param a: vsa vector
    :param b: vsa vector
    :return: cosine distance between a and b, 0.0=exact match.
    """
    if a.validate_operand(b):
        a1, b1 = VsaBase.trunc_vecs_to_same_len(a, b)
        return a.cosine(a1, b1)


def cosine_sim(a, b):
    """
    :param a: vsa vector
    :param b: vsa vector
    :return: cosine similarity between a and b. 1.0=exact match.
    """
    if a.validate_operand(b):
        a1, b1 = VsaBase.trunc_vecs_to_same_len(a, b)
        return a.cosine_sim(a1, b1)


def hsim(a, b):
    """
    Returns hamming similarity between v1 and v2. This is equivalent to (1-hamming_distance)
    :param a:
    :param b:
    :return:
    """
    if a.validate_operand(b):
        a1, b1 = VsaBase.trunc_vecs_to_same_len(a, b)
        return a.hsim(a1, b1)


def hdist(a, b):
    """
    Returns hamming similarity between v1 and v2. This is equivalent to (1-hamming_distance)
    :param a:
    :param b:
    :return:
    """
    if a.validate_operand(b):
        a1, b1 = VsaBase.trunc_vecs_to_same_len(a, b)
        return a.hdist(a1, b1)


# def sum(ndarray, *args, **kwargs):
#     """
#     Maintains vsa_type custom attribute when perfoming numpy.sum()
#     Todo: there is probably a better way than this.
#     """
#     if len(ndarray.shape()) == 1: # If there is only one vector in the list.
#         return ndarray
#     return ndarray[0].sum(ndarray, *args, **kwargs)
