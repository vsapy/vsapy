import numpy as np

from .vsatype import VsaBase, VsaType
from .laiho import Laiho
from .time_stamp import TimeStamp
from .bsc_stats import subvec_mean as bsc_mean


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
    from .bag import BagVec
    if not isinstance(sub_vecs, BagVec):
        num_vecs = sub_vecs
    else:
        num_vecs = sub_vecs.vec_cnt
        if vsa_type is not None and vsa_type != sub_vecs.vsa_type:
            raise ValueError("sub_vecs should be in integer when passing a vsa_type.")
        vsa_type = sub_vecs.vsa_type
        if Laiho.is_laiho_type(sub_vecs):
            bits_per_slot = sub_vecs.myvec.bits_per_slot

    if vsa_type:
        if vsa_type == VsaType.HRR:
            raise NotImplementedError(f'subvec_mean not implemented for type{VsaType.HRR}.')
        elif Laiho.is_laiho_type(vsa_type):
            from .sparse_stats import subvec_mean as snn_mean
            return snn_mean(num_vecs, bits_per_slot, 1)
        else:
            return bsc_mean(num_vecs)

    raise ValueError("vsa_type must be specifed.")


class Real2BinaryLegacy(object):
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
        We then sum along axis=1 which gives us a 'bdim' real-number vector.
        This is then thresholded to produce a binary bit pattern that maintains the distances in the vector space.
        The binary vector produced has an approximately, equal number of 1's and 0's maintaining thus maintaining the
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
        ZZ = np.dot(self.mapper, np.swapaxes(v, 0, 1)).T
        exp = 0.5 * np.sum(v, axis=1)
        var = np.sqrt(np.sum(v * v, axis=1) * 0.25)

        ZZ = ((ZZ.T - exp) / var).T
        return (ZZ >= 0).astype('uint8')


class Real2Binary:
    """
    Random Hyperplane Projection (SimHash / Random Binary Projection).

    Preserves cosine similarity in expectation:
      P[bit differs] = arccos(cos_sim(x,y)) / pi

    Note when converting a 'bank' / database of real-number vectors the same seed MUST be used
    in order to ensure that the semantic vector space distances are maintained.
    Obviously a single run will maintain this since we generate the mapper on initialisation.
    """

    def __init__(self, rdim: int, bdim: int, seed: int | None = None, *, gaussian: bool = False):
        self.rdim = int(rdim)
        self.bdim = int(bdim)

        rng = np.random.default_rng(seed)

        # Hyperplanes: (bdim, rdim)
        if gaussian:
            self.W = rng.standard_normal(size=(self.bdim, self.rdim)).astype(np.float32)
        else:
            # ±1 is fast and works well (zero-mean)
            self.W = rng.choice([-1.0, 1.0], size=(self.bdim, self.rdim)).astype(np.float32)

    def to_bin(self, v: np.ndarray) -> np.ndarray:
        """
        v: shape (rdim,) or (N, rdim)
        returns: shape (bdim,) or (N, bdim) of uint8 {0,1}
        """
        v = np.asarray(v)
        if v.ndim == 1:
            if v.shape[0] != self.rdim:
                raise ValueError(f"Expected v.shape[0]=={self.rdim}, got {v.shape[0]}")
            y = self.W @ v  # (bdim,)
            return (y >= 0).astype(np.uint8)

        if v.ndim == 2:
            if v.shape[1] != self.rdim:
                raise ValueError(f"Expected v.shape[1]=={self.rdim}, got {v.shape[1]}")
            y = v @ self.W.T  # (N, bdim)
            return (y >= 0).astype(np.uint8)

        raise ValueError("v must be 1D or 2D")


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
    return _apply_pairwise("cosine", a, b)


def cosine_sim(a, b):
    """
    :param a: vsa vector
    :param b: vsa vector
    :return: cosine similarity between a and b. 1.0=exact match.
    """
    return _apply_pairwise("cosine_sim", a, b)


def _is_bank(x) -> bool:
    """Return True if x should be treated as a bank (collection) of vectors.

    Supported banks:
      - 2D numpy/VsaBase array shaped (N, D)
      - list/tuple of 1D vectors
    """
    try:
        # VsaBase is an ndarray subclass; treat 2D as a bank.
        if isinstance(x, np.ndarray) and getattr(x, "ndim", 0) == 2:
            return True
    except Exception:
        pass
    return isinstance(x, (list, tuple))


def _bank_len(x) -> int:
    if isinstance(x, np.ndarray) and getattr(x, "ndim", 0) == 2:
        return int(x.shape[0])
    return len(x)


def _bank_get(x, i):
    if isinstance(x, np.ndarray) and getattr(x, "ndim", 0) == 2:
        return x[i]
    return x[i]


# ----------------------------
# Fast NumPy paths for unpacked vectors
# ----------------------------
_EPS = 1e-12


def _stack_bank_truncate(bank):
    """Convert a list/tuple bank into a 2D ndarray by truncating all rows to the same min length."""
    if not isinstance(bank, (list, tuple)):
        return bank
    if len(bank) == 0:
        return np.empty((0, 0), dtype=np.float32)
    # Determine a common length (min) to mimic truncation behaviour safely.
    lens = [int(getattr(v, "shape", [len(v)])[0]) if isinstance(v, np.ndarray) else len(v) for v in bank]
    D = min(lens)
    rows = []
    for v in bank:
        vv = np.asarray(v)
        if vv.ndim != 1:
            raise ValueError("Bank elements must be 1D vectors")
        rows.append(vv[:D])
    return np.stack(rows, axis=0)


def _as_array_bank(x):
    """Return ndarray view for bank inputs (2D). For list/tuple banks, stack+truncate."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return _stack_bank_truncate(x)
    return x


def _fast_hsim_hdist(op_name: str, a_arr: np.ndarray, b_arr: np.ndarray):
    """Vectorized Hamming sim/dist for unpacked discrete vectors."""
    if not (isinstance(a_arr, np.ndarray) and isinstance(b_arr, np.ndarray)):
        return None

    # vec vs bank
    if a_arr.ndim == 1 and b_arr.ndim == 2:
        D = min(a_arr.shape[0], b_arr.shape[1])
        hsim = (b_arr[:, :D] == a_arr[:D]).mean(axis=1, dtype=np.float32)
        return (1.0 - hsim) if op_name == "hdist" else hsim

    if a_arr.ndim == 2 and b_arr.ndim == 1:
        D = min(a_arr.shape[1], b_arr.shape[0])
        hsim = (a_arr[:, :D] == b_arr[:D]).mean(axis=1, dtype=np.float32)
        return (1.0 - hsim) if op_name == "hdist" else hsim

    # bank vs bank
    if a_arr.ndim == 2 and b_arr.ndim == 2:
        D = min(a_arr.shape[1], b_arr.shape[1])
        hsim = (a_arr[:, None, :D] == b_arr[None, :, :D]).mean(axis=2, dtype=np.float32)
        return (1.0 - hsim) if op_name == "hdist" else hsim

    return None


def _fast_cosine(op_name: str, a_arr: np.ndarray, b_arr: np.ndarray):
    """Vectorized cosine distance/similarity."""
    if not (isinstance(a_arr, np.ndarray) and isinstance(b_arr, np.ndarray)):
        return None
    if not (np.issubdtype(a_arr.dtype, np.number) and np.issubdtype(b_arr.dtype, np.number)):
        return None

    def _norm_rows(X):
        return np.linalg.norm(X, axis=1) + _EPS

    # vec vs bank
    if a_arr.ndim == 1 and b_arr.ndim == 2:
        D = min(a_arr.shape[0], b_arr.shape[1])
        v = a_arr[:D].astype(np.float32, copy=False)
        B = b_arr[:, :D].astype(np.float32, copy=False)
        dots = B @ v
        denom = _norm_rows(B) * (np.linalg.norm(v) + _EPS)
        cos_sim = dots / denom
        return (1.0 - cos_sim) if op_name == "cosine" else cos_sim

    if a_arr.ndim == 2 and b_arr.ndim == 1:
        D = min(a_arr.shape[1], b_arr.shape[0])
        A = a_arr[:, :D].astype(np.float32, copy=False)
        v = b_arr[:D].astype(np.float32, copy=False)
        dots = A @ v
        denom = _norm_rows(A) * (np.linalg.norm(v) + _EPS)
        cos_sim = dots / denom
        return (1.0 - cos_sim) if op_name == "cosine" else cos_sim

    # bank vs bank
    if a_arr.ndim == 2 and b_arr.ndim == 2:
        D = min(a_arr.shape[1], b_arr.shape[1])
        A = a_arr[:, :D].astype(np.float32, copy=False)
        B = b_arr[:, :D].astype(np.float32, copy=False)
        dots = A @ B.T
        denom = _norm_rows(A)[:, None] * _norm_rows(B)[None, :]
        cos_sim = dots / denom
        return (1.0 - cos_sim) if op_name == "cosine" else cos_sim

    return None


def _apply_pairwise(op_name: str, a, b):
    """Apply an instance op (hsim/hdist/cosine/cosine_sim/...) across banks.

    Rules:
      - vec vs vec -> scalar
      - vec vs bank -> (N,) ndarray
      - bank vs vec -> (N,) ndarray
      - bank vs bank -> (N, M) ndarray
    """
    a_is_bank = _is_bank(a)
    b_is_bank = _is_bank(b)


    # Fast NumPy paths (unpacked vectors): if either side is a bank, try a vectorized implementation.
    # We also allow list/tuple banks by stacking+truncating to a 2D ndarray.
    a_arr = _as_array_bank(a) if a_is_bank else a
    b_arr = _as_array_bank(b) if b_is_bank else b

    if op_name in ("hsim", "hdist"):
        fast = _fast_hsim_hdist(op_name, np.asarray(a_arr), np.asarray(b_arr))
        if fast is not None:
            return fast

    if op_name in ("cosine", "cosine_sim"):
        fast = _fast_cosine(op_name, np.asarray(a_arr), np.asarray(b_arr))
        if fast is not None:
            return fast

    if (not a_is_bank) and (not b_is_bank):
        if a.validate_operand(b):
            a1, b1 = VsaBase.trunc_vecs_to_same_len(a, b)
            return getattr(a, op_name)(a1, b1)
        return None

    def _one(x, y):
        # Validate on the "x" operand (mirrors existing API behaviour)
        if x.validate_operand(y):
            x1, y1 = VsaBase.trunc_vecs_to_same_len(x, y)
            return getattr(x, op_name)(x1, y1)
        return None

    if a_is_bank and (not b_is_bank):
        n = _bank_len(a)
        out = np.empty((n,), dtype=np.float32)
        for i in range(n):
            out[i] = _one(_bank_get(a, i), b)
        return out

    if (not a_is_bank) and b_is_bank:
        n = _bank_len(b)
        out = np.empty((n,), dtype=np.float32)
        for i in range(n):
            out[i] = _one(a, _bank_get(b, i))
        return out

    # bank vs bank => full similarity/distance matrix
    n = _bank_len(a)
    m = _bank_len(b)
    out = np.empty((n, m), dtype=np.float32)
    for i in range(n):
        ai = _bank_get(a, i)
        for j in range(m):
            out[i, j] = _one(ai, _bank_get(b, j))
    return out


def hsim(a, b):
    """
    Returns hamming similarity between v1 and v2. This is equivalent to (1-hamming_distance)
    :param a:
    :param b:
    :return:
    """
    return _apply_pairwise("hsim", a, b)


def hdist(a, b):
    """
    Returns hamming similarity between v1 and v2. This is equivalent to (1-hamming_distance)
    :param a:
    :param b:
    :return:
    """
    return _apply_pairwise("hdist", a, b)


# def sum(ndarray, *args, **kwargs):
#     """
#     Maintains vsa_type custom attribute when perfoming numpy.sum()
#     Todo: there is probably a better way than this.
#     """
#     if len(ndarray.shape()) == 1: # If there is only one vector in the list.
#         return ndarray
#     return ndarray[0].sum(ndarray, *args, **kwargs)
