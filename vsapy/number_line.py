from __future__ import annotations

from typing import Optional, List
import numpy as np

from .time_stamp import TimeStamp
from . import vsapy as vsa
from .vsatype import VsaBase, VsaType

# ----------------------------
# Base class
# ----------------------------

class ScalarHypervectorEmbeddingBase:
    """
    Base class for scalar->hypervector embeddings.

    Responsibilities:
    - owns vec_dim, vsa_kwargs, make_orthogonal, creation timestamp
    - provides bind/unbind convenience
    - provides fast BSC nearest-neighbour decode over a bank of vectors

    Subclasses implement how the bank is created and how scalar maps to an index.
    """

    def __init__(self, vec_dim: int, vsa_kwargs: dict, creation_data_time_stamp=None):
        self.vec_dim = vec_dim
        self.vsa_kwargs = vsa_kwargs

        if creation_data_time_stamp is None:
            self.creation_data_time_stamp = TimeStamp.get_creation_data_time_stamp()
        else:
            self.creation_data_time_stamp = creation_data_time_stamp

        # Orthogonalizer (domain separator)
        self.make_orthogonal = vsa.randvec(vec_dim, **vsa_kwargs)
        self.default_key = self.make_orthogonal  # New name for better usability.

    # --- binding helpers ---
    def bind(self, a, b):
        return vsa.bind(a, b)

    def unbind(self, a, b):
        return vsa.unbind(a, b)

    # --- distance helpers ---
    @staticmethod
    def _as_bool_array(x) -> np.ndarray:
        # For BSC you should already be bool-ish; force bool for speed.
        return np.asarray(x).astype(bool, copy=False)

    @staticmethod
    def _bsc_hd_bits_bank(bank_bool: np.ndarray, query_bool: np.ndarray) -> np.ndarray:
        # bank_bool: (N,D) bool, query_bool: (D,) bool -> (N,) int hd bits
        return np.count_nonzero(bank_bool ^ query_bool, axis=1)

    @staticmethod
    def _bsc_hsim(a_bool: np.ndarray, b_bool: np.ndarray) -> float:
        return 1.0 - (np.count_nonzero(a_bool ^ b_bool) / a_bool.size)

    def _decode_nearest_index(self, bank_vecs, query_vec) -> int:
        """
        Fast nearest neighbour index for BSC banks.
        """
        if self.vsa_kwargs["vsa_type"] == VsaType.BSC:
            bank_bool = self._as_bool_array(bank_vecs)
            query_bool = self._as_bool_array(query_vec)
            hd_bits = self._bsc_hd_bits_bank(bank_bool, query_bool)
            return int(np.argmin(hd_bits))

        # Tern/TernZero
        r = np.apply_along_axis(np.logical_xor, 1, bank_vecs, query_vec)
        res = np.count_nonzero(r.astype(int), axis=1)
        return int(np.argmin(res))


# ----------------------------
# Linear NumberLine
# ----------------------------
def linear_sequence_gen(max_number: int, start_vec: np.ndarray, rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
    """
    Keeping this as a module definition for backwards compatibility to legacy version.

    """
    if rng is None:
        rng = np.random.default_rng()

    v0 = start_vec.astype(bool, copy=True)
    D = v0.size
    total_flips = D // 2

    if max_number <= 0:
        return [v0]
    if max_number > total_flips:
        raise ValueError(f"max_number={max_number} exceeds D//2={total_flips}.")

    base = total_flips // max_number
    rem = total_flips % max_number
    flips_per_step = np.full(max_number, base, dtype=int)
    if rem:
        flips_per_step[:rem] += 1

    perm = rng.permutation(D)

    seq = [v0]
    v = v0.copy()
    cursor = 0
    for step in range(max_number):
        k = int(flips_per_step[step])
        if k > 0:
            idx = perm[cursor: cursor + k]
            v[idx] = ~v[idx]
            cursor += k
        seq.append(v.copy())
    return seq


class NumberLine(ScalarHypervectorEmbeddingBase):
    """
    Linear number line (min..max inclusive), with optional quantisation into Q steps.

    This is your current behaviour, cleaned up:
    - build bank using linear_sequence_gen
    - number_to_vec binds bank vector with make_orthogonal
    - number_from_vec unbinds and nearest-neighbour decodes
    """

    def __init__(
        self,
        min_number: int,
        max_number: int,
        vec_dim: int,
        vsa_kwargs: dict,
        quantise_interval: int = 0,
        creation_data_time_stamp=None,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(vec_dim=vec_dim, vsa_kwargs=vsa_kwargs, creation_data_time_stamp=creation_data_time_stamp)

        if vsa_kwargs["vsa_type"] not in [VsaType.BSC, VsaType.Tern, VsaType.TernZero]:
            raise NotImplementedError(f"NumberLine is not implemented for type: {VsaType.HRR}.")

        self.min_num = int(min_number)
        self.max_num = int(max_number)
        self.range = self.max_num - self.min_num

        if quantise_interval > vec_dim // 2:
            raise ValueError("parameter 'quantise_interval' must be <= vec_dim // 2")

        # Q = number of representable steps in the bank
        self.Q = min(self.range, vec_dim // 2) if quantise_interval == 0 else int(quantise_interval)

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        # base vector for level generation
        self.zero_vec = vsa.randvec(vec_dim, **vsa_kwargs)
        zvec_bsc = vsa.to_vsa_type(self.zero_vec, new_vsa_type=VsaType.BSC)

        steps = self.Q if self.Q else (self.max_num - self.min_num)
        bank_bsc = linear_sequence_gen(steps, zvec_bsc, rng=self.rng)  # list of bool vecs

        self.number_vecs = vsa.to_vsa_type(
            VsaBase(bank_bsc, vsa_type=VsaType.BSC),
            new_vsa_type=vsa_kwargs["vsa_type"],
        )

    def number_to_vec(self, num: int, *, key=None):
        num = int(num)
        if num < self.min_num or num > self.max_num:
            raise ValueError("Number is out of range.")

        if self.Q:
            norm_n = int((num - self.min_num) / self.range * self.Q)
            base = self.number_vecs[norm_n]
        else:
            base = self.number_vecs[num - self.min_num]

        key = self.make_orthogonal if key is None else key
        return base if key is None else self.bind(base, key)

    def number_from_vec(self, number_vec, *, key=None) -> int:
        key = self.make_orthogonal if key is None else key
        raw_vec = number_vec if key is None else self.unbind(key, number_vec)

        winner = self._decode_nearest_index(self.number_vecs, raw_vec)
        if self.Q:
            return int(winner * self.range / self.Q + self.min_num)
        return int(winner + self.min_num)

    def with_key(self, key):
        """
        Allows us to use the same bank of vecs for x, y number-lines
        key, self.mage_orthogonal/self.default shifts the entire number-line to a new spot in vector space.
        """
        parent = self
        class View:
            def number_to_vec(self, num: int):
                return parent.number_to_vec(num, key=key)
            def number_from_vec(self, vec):
                return parent.number_from_vec(vec, key=key)
        return View()


# ----------------------------
# Circular: folded (flip out then flip back)
# ----------------------------
class CircularNumberLineFolded(ScalarHypervectorEmbeddingBase):
    """
    Circular numberline for degrees (or any periodic scalar).
    Folded construction: move away from start to ~180°, then undo to return to start.

    Pros:
    - simple
    - Plot 1 (from 0°) looks linear-ish to 180°
    Cons:
    - NOT rotationally invariant (Plot 3 differs from Plot 1)
    """

    def __init__(
        self,
        period: float,
        m_steps: int,
        vec_dim: int,
        vsa_kwargs: dict,
        *,
        max_hd: float = 0.5,
        creation_data_time_stamp=None,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(vec_dim=vec_dim, vsa_kwargs=vsa_kwargs, creation_data_time_stamp=creation_data_time_stamp)

        if vsa_kwargs["vsa_type"] != VsaType.BSC:
            raise NotImplementedError("CircularNumberLineFolded currently implemented for BSC only (bool XOR).")

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.period = float(period)
        self.m = int(m_steps)

        start_vec = vsa.randvec(vec_dim, **vsa_kwargs)
        start_bsc = vsa.to_vsa_type(start_vec, new_vsa_type=VsaType.BSC)

        levels = self.circular_level_hypervectors(self.m, np.asarray(start_bsc), max_hd=max_hd, rng=self.rng)
        # store bank excluding duplicated last
        self.levels = np.stack(levels[:-1], axis=0).astype(bool, copy=False)
        self.levels = VsaBase(self.levels, vsa_type=VsaType.BSC)

    def degrees_to_index(self, deg: float) -> int:
        return int(round((deg % self.period) / self.period * self.m)) % self.m

    def index_to_degrees(self, idx: int) -> float:
        return (idx % self.m) * self.period / self.m

    def degrees_to_vec(self, deg: float) -> np.ndarray:
        k = self.degrees_to_index(deg)
        return self.levels[k] ^ self._as_bool_array(self.make_orthogonal)

    def degrees_from_vec(self, vec: np.ndarray) -> int:
        raw = self._as_bool_array(vec) ^ self._as_bool_array(self.make_orthogonal)
        return int(np.argmin(self._bsc_hd_bits_bank(self.levels, raw)))

    @staticmethod
    def circular_level_hypervectors(
        m: int,
        start_vec: np.ndarray,
        *,
        max_hd: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> List[np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        v0 = start_vec.astype(bool, copy=True)
        D = v0.size
        if m < 2:
            raise ValueError("m must be >= 2")

        half1 = m // 2
        half2 = m - half1

        total_flips = int(round(D * max_hd))
        total_flips = min(total_flips, D)

        base = total_flips // half1
        rem = total_flips % half1
        flips_per_step_out = np.full(half1, base, dtype=int)
        if rem:
            flips_per_step_out[:rem] += 1

        perm = rng.permutation(D)
        used = perm[:total_flips]

        chunks = []
        cursor = 0
        for k in flips_per_step_out:
            if k:
                chunks.append(used[cursor:cursor + k])
                cursor += k
            else:
                chunks.append(np.array([], dtype=int))

        seq = [v0]
        v = v0.copy()

        for idx in chunks:
            if idx.size:
                v[idx] = ~v[idx]
            seq.append(v.copy())

        reverse_chunks = list(reversed(chunks))
        for s in range(half2):
            if s < len(reverse_chunks):
                idx = reverse_chunks[s]
                if idx.size:
                    v[idx] = ~v[idx]
            seq.append(v.copy())

        # closure
        assert np.array_equal(seq[0], seq[-1])
        return seq


# ----------------------------
# Circular: true rotational (order-m permutation)
# ----------------------------
class CircularNumberLinePhaseProjected(ScalarHypervectorEmbeddingBase):
    """
    circular geometry built in real space (Fourier/phase embedding),
    then projected to BSC using random hyperplanes (SimHash-style).

    This produces smooth, rotationally invariant similarity vs angular difference.
    """

    def __init__(
        self,
        period: float,
        m_steps: int,
        vec_dim: int,
        vsa_kwargs: dict,
        *,
        rdim: int = 2, # real embedding dimension (must be even number)
        n_harmonics: Optional[int] = None, # if None, use rdim//2
        creation_data_time_stamp=None,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(vec_dim=vec_dim, vsa_kwargs=vsa_kwargs, creation_data_time_stamp=creation_data_time_stamp)

        if vsa_kwargs["vsa_type"] != VsaType.BSC:
            raise NotImplementedError("PhaseProjectedBSC currently implemented for BSC output only.")

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.period = float(period)
        self.m = int(m_steps)
        self.D = int(vec_dim)

        # --- real embedding config ---
        if rdim % 2 != 0:
            raise ValueError("rdim must be even (cos/sin pairs).")
        self.rdim = int(rdim)

        H = (rdim // 2) if n_harmonics is None else int(n_harmonics)
        if 2 * H != rdim:
            raise ValueError("rdim must equal 2*n_harmonics.")
        self.H = H

        # Random hyperplanes for SimHash-style projection.
        # Must be zero-mean isotropic; Gaussian is preferred,
        # especially when rdim is small (e.g. 2).
        # Shape: (D, rdim)
        self.W = self.rng.standard_normal(size=(self.D, self.rdim)).astype(np.float32)

        # Precompute bank of BSC vectors for k=0..m-1
        thetas = (np.arange(self.m) / self.m) * (2.0 * np.pi)  # radians
        R = self._phase_embed(thetas)                          # (m, rdim)
        Z = (self.W @ R.T).T                                   # (m, D)
        self.levels = (Z >= 0).astype(bool, copy=False)
        self.levels = VsaBase(self.levels, vsa_type=VsaType.BSC)


    def degrees_to_index(self, deg: float) -> int:
        return int(round((deg % self.period) / self.period * self.m)) % self.m

    def degrees_to_vec(self, deg: float) -> np.ndarray:
        k = self.degrees_to_index(deg)
        return self.levels[k] ^ self._as_bool_array(self.make_orthogonal)

    def degrees_from_vec(self, vec: np.ndarray) -> int:
        raw = self._as_bool_array(vec) ^ self._as_bool_array(self.make_orthogonal)
        return int(np.argmin(self._bsc_hd_bits_bank(self.levels, raw)))

    @property
    def levels_closed(self) -> np.ndarray:
        return np.concatenate([self.levels, self.levels[:1]], axis=0)

    def _phase_embed(self, theta: np.ndarray) -> np.ndarray:
        """
        theta: shape (m,) radians
        returns: shape (m, 2H) with [cos(hθ), sin(hθ)] for h=1..H
        """
        theta = theta.reshape(-1, 1)  # (m,1)
        hs = np.arange(1, self.H + 1, dtype=np.float32).reshape(1, -1)  # (1,H)
        ang = theta.astype(np.float32) * hs  # (m,H)

        cos_part = np.cos(ang)
        sin_part = np.sin(ang)
        return np.concatenate([cos_part, sin_part], axis=1)  # (m, 2H)

