"""
vsapy.flight_radar_vsa
----------------------

Searchable VSA encoding for FlightRadar24 JSON snapshots.

Design goals
- Search by key presence: "has statusDetails.heading"
- Search by key=value: "airline.code.icao == DLH"
- Optional numeric similarity (lat/lng/alt/hd/spd) via custom numeric encoder hooks
- Hierarchical "subtree" vectors for descent/refinement (e.g. flightHistory.arrival)

Compatible with VsaType.BSC (binary spatter codes) and VsaType.Tern (bipolar ±1).
For BSC, weighting is implemented by repetition.
For Tern, weighting is implemented by scalar accumulation then sign-normalisation.

You provide:
- vsa_tok: VsaTokenizer (holds shared symbol_dict / alphabet)
- optional numeric_encoders: map from tuple(path_keys) -> callable(value)->VsaBase
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import DefaultDict
from collections import defaultdict
from itertools import islice
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .cspvec import CSPvec
from .vsapy import bind, normalize, unbind, hsim
from .bag import BagVec
from .vsatype import VsaBase, VsaType

Path = Tuple[str, ...]
NumericEncoder = Callable[[Union[int, float]], VsaBase]
ValueVectoriser = Callable[[str], VsaBase]


def chunk_word_vector(self, word):
    lettervecs = [self.symbol_dict[a] for a in word]

    cnk = CSPvec(word, lettervecs, self.role_vecs)
    if self.usechunksforwords:
        self.linecheck.append(word)  # record the word for verification

    return cnk

def match_letter(invec: VsaBase, pindex: int, cleanup: Dict[str, VsaBase], perm_vecs: List[VsaBase]) -> (str, VsaBase):
    vv = np.roll(unbind(invec, np.roll(perm_vecs[pindex], -1 * pindex)), -1)
    max_sim = 0.5
    best_match = None
    for k, v in islice(cleanup.items(), 1, None):
        hs = hsim(vv, v)
        if hs > max_sim:
            max_sim = hs
            best_match = k

    if max_sim < 0.53:
        return "", vv
    return best_match, vv

def unbind_value(invec: VsaBase, pindex: int, cleanup: Dict[str, VsaBase], perm_vecs: List[VsaBase]):
    val = ""
    for i in range(len(perm_vecs)):
        l, invec = match_letter(invec, i, cleanup, perm_vecs)
        if not l:
            continue
        val += l

    return val

def keyname2vsa(name: str, vec_alphabet: Dict[str, VsaBase]) -> VsaBase:
    """
    Deterministically create a unique role vector from a string, given a shared alphabet dict.
    (Derived from tests/json2vsa.py.)
    """
    n = name.lower()
    if not n:
        n = "x"
    v = vec_alphabet[n[0]]
    shift = 0
    for c in n[1:]:
        shift += 1
        v = bind(v, np.roll(vec_alphabet[c], shift))
    return v


@dataclass(frozen=True)
class MarkerRoles:
    # top-level partitions
    role_id: VsaBase
    role_var: VsaBase

    # feature types
    key_present: VsaBase
    key_value: VsaBase
    key_num: VsaBase

    # hierarchy / arrays
    subtree_mark: VsaBase
    trail_item_mark: VsaBase
    item_mark: VsaBase


def _mk_roles(vsa_tok, names: Dict[str, str]) -> MarkerRoles:
    a = vsa_tok.symbol_dict

    def k(n: str) -> VsaBase:
        return keyname2vsa(names[n], a)

    return MarkerRoles(
        role_id=k("role_id"),
        role_var=k("role_var"),
        key_present=k("key_present"),
        key_value=k("key_value"),
        key_num=k("key_num"),
        subtree_mark=k("subtree_mark"),
        trail_item_mark=k("trail_item_mark"),
        item_mark=k("item_mark"),
    )


class WeightedAccumulator:
    """
    Accumulate vectors with optional weights.

    - For BSC: implemented by repetition (integer reps).
    - For Tern: accumulate in float then normalize by sign (Tern.normalize()).
    """

    def __init__(self, vsa_type: VsaType):
        self.vsa_type = vsa_type
        self._vecs: List[VsaBase] = []
        self._float_sum: Optional[np.ndarray] = None
        self._template: Optional[VsaBase] = None
        self._effective_cnt: float = 0.0

    def add(self, v: VsaBase, weight: float = 1.0) -> None:
        if weight <= 0:
            return

        if self.vsa_type == VsaType.BSC:
            reps = int(round(weight))
            reps = max(1, reps)
            for _ in range(reps):
                self._vecs.append(v)
            self._effective_cnt += reps
            return

        # Tern / others: float accumulate
        if self._template is None:
            self._template = v
            self._float_sum = np.zeros_like(np.asarray(v), dtype=np.float32)

        assert self._float_sum is not None
        self._float_sum += (float(weight) * np.asarray(v, dtype=np.float32))
        self._effective_cnt += float(weight)

    def extend(self, vs: Iterable[VsaBase], weight: float = 1.0) -> None:
        for v in vs:
            self.add(v, weight=weight)

    def to_vec(self) -> VsaBase:
        if self.vsa_type == VsaType.BSC:
            if not self._vecs:
                raise ValueError("Accumulator is empty")
            # BagVec.bundle returns (rawvec, vec_cnt, norm_vec)
            _, _, norm = BagVec.bundle(
                self._vecs,
                vec_cnt=int(self._effective_cnt) if self._effective_cnt else -1,
            )
            return norm

        if self._template is None or self._float_sum is None:
            raise ValueError("Accumulator is empty")

        sv = self._template.copy()
        sv = sv.astype(np.float32, copy=False)
        sv[:] = self._float_sum
        return normalize(sv, None)

# def decode_wordvector_gb(
#     invec: VsaBase,
#     *,
#     cleanup: Dict[str, VsaBase],
#     max_len: int = 32,
#     # min_sim: float = 0.53,
#     min_sim: float = 0.5125,
#     allowed_chars: Optional[Iterable[str]] = None,
#     stop_after_misses: int = 2,
# ) -> str:
#     """
#     Attempt to invert createWordVector_GB():
#
#         createWordVector_GB(word):
#             for shift=1..len(word):
#                 letter_vecs.append(roll(symbol[c], shift))
#             return bundle(letter_vecs)
#
#     Decode approach:
#       For each position i=1..max_len:
#         vv = roll(invec, -i)  # bring i-th letter back to base orientation
#         cleanup-match vv against symbol vectors (optionally restricted alphabet).
#       Stop after N consecutive misses.
#
#     Notes:
#     - This is approximate cleanup under superposition noise.
#     - Works best for short strings and constrained alphabets.
#     """
#
#     # Normalize the input estimate to sharpen similarity
#     vin = normalize(invec.copy(), None)
#
#     if allowed_chars is not None:
#         allowed_set = set(allowed_chars)
#         items = [(k, v) for k, v in cleanup.items() if k in allowed_set]
#     else:
#         items = list(cleanup.items())
#
#     out = []
#     misses = 0
#
#     # Positions start at 1 because encoder starts shift at 1
#     for i in range(1, max_len + 1):
#         vv = np.roll(vin, -i)
#
#         best_k = ""
#         best_sim = -1.0
#         for k, v in items:
#             hs = hsim(vv, v)
#             if hs > best_sim:
#                 best_sim = hs
#                 best_k = k
#
#         if best_sim >= min_sim:
#             out.append(best_k)
#             misses = 0
#         else:
#             misses += 1
#             if misses >= stop_after_misses:
#                 break
#
#     return "".join(out)

def decode_wordvector_gb(
    invec: VsaBase,
    *,
    cleanup: Dict[str, VsaBase],
    max_len: int = 64,
    min_sim: float = 0.53,
    allowed_chars: Optional[Iterable[str]] = None,
    stop_after_misses: int = 2,
) -> str:
    """
    Approximate inverse of createWordVector_GB(word):

        wordvec = bundle( roll(symbol[c1],1), roll(symbol[c2],2), ... )

    Decode:
      for i=1..max_len:
        vv = roll(invec, -i)
        find best cleanup match among allowed chars
        if best_sim < min_sim => miss
           stop after N consecutive misses
        else append char

    Works best for short-ish strings and restricted alphabets.
    """
    vin = normalize(invec.copy(), None)

    if allowed_chars is not None:
        allowed = set(allowed_chars)
        items = [(k, v) for k, v in cleanup.items() if k in allowed]
    else:
        items = list(cleanup.items())

    out = []
    misses = 0

    for i in range(1, max_len + 1):
        vv = np.roll(vin, -i)

        best_k = ""
        best_sim = -1.0
        for k, v in items:
            hs = hsim(vv, v)
            if hs > best_sim:
                best_sim = hs
                best_k = k

        if best_sim >= min_sim:
            out.append(best_k)
            misses = 0
        else:
            misses += 1
            if misses >= stop_after_misses:
                break

    return "".join(out)

def decode_wordvector_gb_from_unbound(
    invec_unbound: VsaBase,
    *,
    cleanup: Dict[str, VsaBase],
    max_len: int = 64,
    min_sim: float = 0.53,
    allowed_chars: Optional[Iterable[str]] = None,
    stop_after_misses: int = 4,
) -> str:
    """
    Decode word vectors AFTER handle has been removed:

        unbind(KV, handle) ~= sum_i roll(letter_i, i)

    So decode per position:
        vv_i = roll(unbound, -i) ~= letter_i + noise
    """
    vin = normalize(invec_unbound.copy(), None)

    if allowed_chars is not None:
        allowed = set(allowed_chars)
        items = [(k, v) for k, v in cleanup.items() if k in allowed]
    else:
        items = list(cleanup.items())

    out = []
    misses = 0

    for i in range(1, max_len + 1):
        vv = np.roll(vin, -i)

        best_k = ""
        best_sim = -1.0
        for kch, vch in items:
            hs = hsim(vv, vch)
            if hs > best_sim:
                best_sim = hs
                best_k = kch

        if best_sim >= min_sim:
            out.append(best_k)
            misses = 0
        else:
            misses += 1
            if misses >= stop_after_misses:
                break

    return "".join(out)

class FlightRadarVsaEncoder:
    """
    Searchable encoder for FlightRadar24 JSON snapshots.

    Feature vectors emitted per leaf:
      - K(path): key presence marker
      - KV(path, value): key/value binding
      - KNUM(path, num): optional numeric encoding via hooks

    Hierarchy:
      - SUB(prefix): subtree bags for selected prefixes

    Arrays:
      - trail[] encoded as item-bound mini-bags using ITEM(time_bucket or index)

    Output:
      AIRCRAFT = (ROLE_ID * FIXED_BAG) + (ROLE_VAR * VAR_BAG)
    """

    DEFAULT_ROLE_NAMES = {
        "role_id": "roleid",
        "role_var": "rolevar",
        "key_present": "keypresent",
        "key_value": "keyvalue",
        "key_num": "keynum",
        "subtree_mark": "subtreemark",
        "trail_item_mark": "trailitemmark",
        "item_mark": "itemmark",
    }

    def __init__(
        self,
        vsa_tok,
        *,
        vsa_type: VsaType = VsaType.BSC,
        value_vectoriser: Optional[ValueVectoriser] = None,
        numeric_encoders: Optional[Dict[Path, NumericEncoder]] = None,
        subtree_prefixes: Optional[Sequence[Path]] = None,
        var_prefixes: Optional[Sequence[Path]] = None,
        trail_ts_bucket_seconds: int = 60,
    ):
        self.vsa_tok = vsa_tok
        self.vsa_type = vsa_type
        self.value_vectoriser = value_vectoriser or (lambda s: vsa_tok.createWordVector_GB(s))
        self.numeric_encoders = numeric_encoders or {}

        # Telemetry-ish blocks
        self.var_prefixes: List[Path] = list(
            var_prefixes
            or [
                ("location",),
                ("statusDetails",),
                ("trail",),
            ]
        )

        # Subtrees to bind (for “descent/refinement”)
        self.subtree_prefixes: List[Path] = list(
            subtree_prefixes
            or [
                ("identification",),
                ("aircraft",),
                ("airline",),
                ("status",),
                ("flightHistory", "departure"),
                ("flightHistory", "arrival"),
                ("location",),
                ("statusDetails",),
                ("trail",),
            ]
        )

        self.trail_ts_bucket_seconds = int(trail_ts_bucket_seconds)
        self.roles = _mk_roles(vsa_tok, self.DEFAULT_ROLE_NAMES)

    # ------------------ low-level builders ------------------

    def key_vec(self, key: str) -> VsaBase:
        return keyname2vsa(key, self.vsa_tok.symbol_dict)

    def chain_vec(self, path: Path) -> VsaBase:
        assert len(path) >= 1
        v = self.key_vec(path[0])
        for i, k in enumerate(path[1:], start=1):
            v = bind(v, np.roll(self.key_vec(k), i))
        return v

    def _is_var_path(self, path: Path) -> bool:
        for pfx in self.var_prefixes:
            if path[: len(pfx)] == pfx:
                return True
        return False

    def vec_key_present(self, path: Path) -> VsaBase:
        return bind(self.roles.key_present, self.chain_vec(path))

    # def vec_key_value(self, path: Path, value: Any) -> VsaBase:
    #     chain = self.chain_vec(path)
    #     # late-bind characters to the key-chain
    #     vv = self.vsa_tok.createWordVector_GB(str(value), key=chain)
    #     return bind(bind(self.roles.key_value, chain), vv)

    def vec_key_value(self, path: Path, value: Any) -> VsaBase:
        """
        Late-bind characters to the per-leaf handle:
            H = KEY_VALUE * chain(path)
            KV = bundle( H*roll(char1,1), H*roll(char2,2), ... )
        """
        handle = bind(self.roles.key_value, self.chain_vec(path))
        return self.vsa_tok.createWordVector_GB(str(value), key=handle)


    def vec_key_num(self, path: Path, num: Union[int, float]) -> Optional[VsaBase]:
        enc = self.numeric_encoders.get(path)
        if enc is None:
            return None
        nv = enc(num)
        return bind(bind(self.roles.key_num, self.chain_vec(path)), nv)

    def vec_item_role(self, item_id: str) -> VsaBase:
        return bind(self.roles.item_mark, self.key_vec(item_id))

    # ------------------ traversal / emission ------------------

    def _emit_leaf(self, path: Path, value: Any) -> List[VsaBase]:
        """
        Emit vectors for a single leaf (path,value).
        Always emits key-presence and key-value.
        Optionally emits numeric encoding if a numeric encoder exists for this exact path.
        """
        out = [self.vec_key_present(path), self.vec_key_value(path, value)]

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            vn = self.vec_key_num(path, value)
            if vn is not None:
                out.append(vn)
        return out

    def _flatten_emit(self, obj: Any, prefix: Path = ()) -> List[Tuple[Path, Any, List[VsaBase]]]:
        """
        Flatten and emit vectors for all leaves.
        Special-cases trail[] when encountered as dict["trail"].
        Returns list of (leaf_path, value, vectors_emitted_for_leaf).
        """
        emitted: List[Tuple[Path, Any, List[VsaBase]]] = []

        if isinstance(obj, dict):
            for k, v in obj.items():
                p = prefix + (str(k),)
                if p == ("trail",) and isinstance(v, list):
                    emitted.extend(self._emit_trail_leaf_debug(v))
                else:
                    emitted.extend(self._flatten_emit(v, p))
            return emitted

        if isinstance(obj, list):
            for item in obj:
                emitted.extend(self._flatten_emit(item, prefix))
            return emitted

        vecs = self._emit_leaf(prefix, obj)
        emitted.append((prefix, obj, vecs))
        return emitted

    def _emit_trail_leaf_debug(self, trail_list: List[Any]) -> List[Tuple[Path, Any, List[VsaBase]]]:
        """
        Debug-only “leaf list” for trail: emits canonical ("trail", key) leaves,
        while the real trail encoding is done as item-bags in _encode_trail_bag().
        """
        emitted: List[Tuple[Path, Any, List[VsaBase]]] = []
        for item in trail_list:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                path = ("trail", str(k))
                vecs = self._emit_leaf(path, v)
                emitted.append((path, v, vecs))
        return emitted

    # ------------------ high-level encode ------------------
    def decode_string_value(
            self,
            aircraft_vec: VsaBase,
            path: Path,
            *,
            max_len: int = 32,
            # min_sim: float = 0.53,
            min_sim: float = 0.510,
            allowed_chars: Optional[Iterable[str]] = None,
            stop_after_misses: int = 2,
    ) -> str:
        """
        Given a full aircraft_vec and a key path, try to recover the string value.

        Steps:
          1) Choose partition: ROLE_ID or ROLE_VAR based on path prefix
          2) Unbind partition role to get approximate partition bag
          3) Unbind the KV handle: (KEY_VALUE * chain(path))
          4) Decode value vector using inverse of createWordVector_GB

        Returns decoded string (may be partial/empty if cleanup fails).
        """

        # 1) select partition role
        part_role = self.roles.role_var if self._is_var_path(path) else self.roles.role_id

        # 2) approximate partition bag
        part_bag = unbind(aircraft_vec, part_role)

        # 3) unbind KV handle to estimate the value vector
        kv_handle = bind(self.roles.key_value, self.chain_vec(path))
        val_hat = unbind(part_bag, kv_handle)

        # 4) decode using GB inverse cleanup
        cleanup = self.vsa_tok.symbol_dict

        # Sensible default alphabet restriction if caller doesn't supply one:
        # (cuts false positives dramatically)
        if allowed_chars is None:
            allowed_chars = (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
                " -_:/."
            )

        return decode_wordvector_gb(
            val_hat,
            cleanup=cleanup,
            max_len=max_len,
            min_sim=min_sim,
            allowed_chars=allowed_chars,
            stop_after_misses=stop_after_misses,
        )

    def encode(self, fr24_json: Dict[str, Any], *, include_debug: bool = True) -> Tuple[VsaBase, Optional[Dict[str, Any]]]:
        """
        Encode a FlightRadar24 JSON snapshot to a single aircraft vector.

        Returns (aircraft_vec, debug_dict_or_None).
        """
        leaves = self._flatten_emit(fr24_json, ())

        fixed_acc = WeightedAccumulator(self.vsa_type) # Fixed part
        var_acc = WeightedAccumulator(self.vsa_type)  # Variable Part

        for path, _, vecs in leaves:
            # Decides if vector(s) go in var_acc or fixed_acc
            target = var_acc if self._is_var_path(path) else fixed_acc
            target.extend(vecs, weight=1.0)

        subtree_debug = {}
        for pfx in self.subtree_prefixes:
            subtree_vec = self._encode_subtree(fr24_json, pfx)
            if subtree_vec is None:
                continue
            subtree_binding = bind(bind(self.roles.subtree_mark, self.chain_vec(pfx)), subtree_vec)
            target = var_acc if self._is_var_path(pfx) else fixed_acc
            target.add(subtree_binding)

            if include_debug:
                subtree_debug["/".join(pfx)] = subtree_vec

        fixed_bag = fixed_acc.to_vec()
        var_bag = var_acc.to_vec()

        # Combine the two partitions
        aircraft_vec = BagVec.bundle(
            [
                bind(self.roles.role_id, fixed_bag),
                bind(self.roles.role_var, var_bag),
            ]
        )[2]

        if not include_debug:
            return aircraft_vec, None

        debug = {
            "leaf_count": len(leaves),
            "subtrees": list(subtree_debug.keys()),
        }
        return aircraft_vec, debug

    def _get_subtree(self, obj: Any, pfx: Path) -> Optional[Any]:
        cur = obj
        for k in pfx:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    def _encode_subtree(self, fr24_json: Dict[str, Any], pfx: Path) -> Optional[VsaBase]:
        sub = self._get_subtree(fr24_json, pfx)
        if sub is None:
            return None

        if pfx == ("trail",) and isinstance(sub, list):
            return self._encode_trail_bag(sub)

        leaves = self._flatten_emit(sub, pfx)
        acc = WeightedAccumulator(self.vsa_type)
        for _, _, vecs in leaves:
            acc.extend(vecs, weight=1.0)
        return acc.to_vec()

    def _encode_trail_bag(self, trail_list: List[Any]) -> VsaBase:
        acc = WeightedAccumulator(self.vsa_type)

        for i, item in enumerate(trail_list):
            if not isinstance(item, dict):
                continue

            # Prefer time-bucket if ts exists, else index
            item_id = f"i{i}"
            ts = item.get("ts")
            if isinstance(ts, (int, float)):
                bucket = int(ts // self.trail_ts_bucket_seconds)
                item_id = f"t{bucket}"

            item_role = self.vec_item_role(item_id)

            item_acc = WeightedAccumulator(self.vsa_type)
            for k, v in item.items():
                path = ("trail", str(k))
                item_acc.extend(self._emit_leaf(path, v), weight=1.0)

            item_bag = item_acc.to_vec()
            trail_item_vec = bind(bind(self.roles.trail_item_mark, item_role), item_bag)
            acc.add(trail_item_vec, weight=1.0)

        return acc.to_vec()

    # ------------------ query builders ------------------

    def query_key_present(self, path: Path) -> VsaBase:
        inner = self.vec_key_present(path)
        role = self.roles.role_var if self._is_var_path(path) else self.roles.role_id
        return bind(role, inner)

    def query_key_value(self, path: Path, value: Any) -> VsaBase:
        inner = self.vec_key_value(path, value)
        role = self.roles.role_var if self._is_var_path(path) else self.roles.role_id
        return bind(role, inner)

    def query_numeric(self, path: Path, num: Union[int, float]) -> VsaBase:
        inner = self.vec_key_num(path, num)
        if inner is None:
            raise KeyError(f"No numeric encoder configured for path={path}")
        role = self.roles.role_var if self._is_var_path(path) else self.roles.role_id
        return bind(role, inner)

    def query_weighted(self, parts: Sequence[Tuple[VsaBase, float]]) -> VsaBase:
        """
        Weighted query composition.

        parts: [(q1, w1), (q2, w2), ...]
        For BSC: weights are rounded to repetition counts.
        For Tern: weights are scalar weights.

        Returns a single query vector.
        """
        acc = WeightedAccumulator(self.vsa_type)
        for q, w in parts:
            acc.add(q, weight=w)
        return acc.to_vec()



@dataclass(frozen=True)
class ChunkRoles:
    chunk_mark: VsaBase
    chunk_index_mark: VsaBase

def _mk_chunk_roles(vsa_tok) -> ChunkRoles:
    a = vsa_tok.symbol_dict
    return ChunkRoles(
        chunk_mark=keyname2vsa("chunkmark", a),
        chunk_index_mark=keyname2vsa("chunkindex", a),
    )

class FlightRadarChunkedEncoder(FlightRadarVsaEncoder):
    """
    Chunked, descendable encoding.

    Returns a list of chunk vectors instead of a single aircraft_vec.
    Each chunk is labeled by (prefix, chunk_index) so you can descend.
    """

    def __init__(
        self,
        vsa_tok,
        *,
        vsa_type: VsaType = VsaType.BSC,
        value_vectoriser: Optional[ValueVectoriser] = None,
        numeric_encoders: Optional[Dict[Path, NumericEncoder]] = None,
        var_prefixes: Optional[Sequence[Path]] = None,
        trail_ts_bucket_seconds: int = 60,
        # new params:
        chunk_size: int = 80,
        # which prefixes get their own descendable chunk-groups:
        chunk_prefixes: Optional[Sequence[Path]] = None,
        include_key_present: bool = True,
        include_key_value: bool = True,
        include_numeric: bool = True,
    ):
        super().__init__(
            vsa_tok,
            vsa_type=vsa_type,
            value_vectoriser=value_vectoriser,
            numeric_encoders=numeric_encoders,
            subtree_prefixes=None,  # IMPORTANT: we do NOT do subtree duplication in chunked mode
            var_prefixes=var_prefixes,
            trail_ts_bucket_seconds=trail_ts_bucket_seconds,
        )
        self.chunk_roles = _mk_chunk_roles(vsa_tok)
        self.chunk_size = int(chunk_size)

        self.include_key_present = include_key_present
        self.include_key_value = include_key_value
        self.include_numeric = include_numeric

        # These are the “descend roots” we produce separate chunk streams for.
        self.chunk_prefixes: List[Path] = list(
            chunk_prefixes
            or [
                ("identification",),
                ("aircraft",),
                ("owner",),
                ("airline",),
                ("status",),
                ("flightHistory", "departure"),
                ("flightHistory", "arrival"),
                ("airport",),
                ("time",),
                ("location",),
                ("statusDetails",),
                ("trail",),
            ]
        )

    # def _emit_leaf(self, path: Path, value: Any) -> List[VsaBase]:
    #     """Override to allow toggling which emissions count toward chunk capacity."""
    #     out: List[VsaBase] = []
    #     if self.include_key_present:
    #         out.append(self.vec_key_present(path))
    #     if self.include_key_value:
    #         out.append(self.vec_key_value(path, value))
    #     if self.include_numeric and isinstance(value, (int, float)) and not isinstance(value, bool):
    #         vn = self.vec_key_num(path, value)
    #         if vn is not None:
    #             out.append(vn)
    #     return out

    def _emit_leaf(self, path: Path, value: Any) -> List[VsaBase]:
        """
        Emit vectors for a single leaf (path,value).
          - key presence: one vector
          - key/value: for strings => many per-letter vectors (late-bundled)
                      for non-strings => one vector (current fallback)
          - optional numeric encoding: one vector if configured
        """
        out: List[VsaBase] = []

        # key presence
        out.append(self.vec_key_present(path))

        # key/value (late-bundled for strings)
        out.extend(self.vec_key_value_terms(path, value))

        # numeric (optional)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            vn = self.vec_key_num(path, value)
            if vn is not None:
                out.append(vn)

        return out

    def _chunk_index_vec(self, idx: int) -> VsaBase:
        # deterministic per-index role
        return bind(self.chunk_roles.chunk_index_mark, self.key_vec(f"c{idx}"))

    def _label_chunk(self, prefix: Path, chunk_idx: int, chunk_vec: VsaBase) -> VsaBase:
        label = bind(self.chunk_roles.chunk_mark, self.chain_vec(prefix))
        label = bind(label, self._chunk_index_vec(chunk_idx))

        # IMPORTANT: bundle label + chunk_vec, don't bind
        if self.vsa_type == VsaType.BSC:
            return BagVec.bundle([label, chunk_vec])[2]
        else:
            acc = WeightedAccumulator(self.vsa_type)
            acc.add(label, weight=0.2)
            acc.add(chunk_vec, weight=1.0)
            return acc.to_vec()

    # def _label_chunk(
    #         self,
    #         chunk_content: VsaBase,
    #         group_prefix: Path,
    #         chunk_idx: int,
    # ) -> VsaBase:
    #
    #     label = bind(self.roles.chunk_mark, self.chain_vec(group_prefix))
    #
    #     if self.vsa_type == VsaType.BSC:
    #         # For BSC we cannot scale — so just repeat content instead
    #         # Give content more weight than label
    #         return BagVec.bundle([label] + [chunk_content] * 4)[2]
    #
    #     # For Tern (or Laiho variants)
    #     acc = WeightedAccumulator(self.vsa_type)
    #
    #     acc.add(label, weight=0.2)  # small weight
    #     acc.add(chunk_content, weight=1.0)  # dominant
    #
    #     return acc.to_vec()

    def encode_chunks(
        self,
        fr24_json: Dict[str, Any],
        *,
        include_debug: bool = True,
    ) -> Tuple[List[VsaBase], Optional[Dict[str, Any]]]:
        """
        Returns list of descendable chunk vectors.

        Chunking rule: keep adding emitted vectors until count would exceed chunk_size,
        then start a new chunk.
        """
        chunks: List[VsaBase] = []
        debug: Dict[str, Any] = {"chunk_groups": {}, "chunk_size": self.chunk_size} if include_debug else {}

        # Group leaf emissions by chosen subtree prefixes so you can “descend”.
        grouped: DefaultDict[Path, List[VsaBase]] = defaultdict(list)

        # Helper: assign any leaf path to the deepest matching chunk_prefix
        def best_prefix(path: Path) -> Path:
            best = ()
            for pfx in self.chunk_prefixes:
                if path[: len(pfx)] == pfx and len(pfx) > len(best):
                    best = pfx
            return best if best else ("__root__",)

        # Walk JSON; trail special-case still emits debug leaves, but we ALSO want proper trail item encoding.
        leaves = self._flatten_emit(fr24_json, ())

        for path, value, vecs in leaves:
            pfx = best_prefix(path)
            grouped[pfx].extend(vecs)

        # Replace the trail group with the *real* trail item-bag encoding if present
        trail = fr24_json.get("trail")
        if isinstance(trail, list):
            # Encode each trail item as its own “leaf-like” packet to keep it queryable
            # (trail bag itself can be large, so we chunk the items)
            trail_item_vecs: List[VsaBase] = []
            for i, item in enumerate(trail):
                if not isinstance(item, dict):
                    continue
                # build item bag
                item_acc = WeightedAccumulator(self.vsa_type)
                for k, v in item.items():
                    path = ("trail", str(k))
                    item_acc.extend(self._emit_leaf(path, v), weight=1.0)
                item_bag = item_acc.to_vec()

                # item role by ts bucket or index (same as previous design)
                item_id = f"i{i}"
                ts = item.get("ts")
                if isinstance(ts, (int, float)):
                    bucket = int(ts // self.trail_ts_bucket_seconds)
                    item_id = f"t{bucket}"
                item_role = self.vec_item_role(item_id)

                trail_item_vec = bind(bind(self.roles.trail_item_mark, item_role), item_bag)
                trail_item_vecs.append(trail_item_vec)

            grouped[("trail",)] = trail_item_vecs

        # Now chunk each group separately and label by prefix + chunk index
        for pfx, vec_list in grouped.items():
            # chunk into bundles of size <= chunk_size
            chunk_idx = 0
            i = 0
            while i < len(vec_list):
                part = vec_list[i : i + self.chunk_size]
                # bundle this chunk
                if self.vsa_type == VsaType.BSC:
                    _, _, cvec = BagVec.bundle(part)
                else:
                    acc = WeightedAccumulator(self.vsa_type)
                    acc.extend(part, weight=1.0)
                    cvec = acc.to_vec()

                labeled = self._label_chunk(pfx, chunk_idx, cvec)
                chunks.append(labeled)

                chunk_idx += 1
                i += self.chunk_size

            if include_debug:
                debug["chunk_groups"]["/".join(pfx)] = {
                    "items": len(vec_list),
                    "chunks": chunk_idx,
                }

        return chunks, (debug if include_debug else None)

    def query_key_present_raw(self, path: Path) -> VsaBase:
        # unchanged: this is exactly what is emitted
        return self.vec_key_present(path)

    # def query_key_value_raw(self, path: Path, value: Any) -> VsaBase:
    #     # must match new emission exactly
    #     return self.vec_key_value(path, value)
    #
    # def find_best_chunk_for_path(
    #     self,
    #     chunks: Sequence[VsaBase],
    #     path: Path,
    # ) -> Tuple[int, float]:
    #     """
    #     Return (best_chunk_index, best_similarity) using key-presence query.
    #     """
    #     q = self.vec_key_present(path)  # RAW: matches what is in chunk content
    #     best_i = -1
    #     best_s = -1.0
    #     for i, c in enumerate(chunks):
    #         s = hsim(q, c)
    #         if s > best_s:
    #             best_s = s
    #             best_i = i
    #     return best_i, best_s

    def query_key_value_raw(self, path: Path, value: Any) -> VsaBase:
        # build query as a bag of the same emitted terms
        terms = self.vec_key_value_terms(path, str(value) if not isinstance(value, str) else value)

        acc = WeightedAccumulator(self.vsa_type)
        for t in terms:
            acc.add(t, weight=1.0)
        return acc.to_vec()

    def decode_string_value_from_chunks(
            self,
            chunks: Sequence[VsaBase],
            path: Path,
            *,
            max_len: int = 64,
            min_sim_key: float = 0.55,
            min_sim_char: float = 0.53,
            allowed_chars: Optional[Iterable[str]] = None,
            stop_after_misses: int = 2,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        1) find best chunk for key presence
        2) unbind KV handle from that chunk to estimate value vector
        3) decode string via miss-based GB decoder

        Returns (decoded_string, debug_info)
        """
        chunk_i, key_sim = self.find_best_chunk_for_path(chunks, path)

        if chunk_i < 0 or key_sim < min_sim_key:
            return "", {"ok": False, "reason": "key_not_found", "best_chunk": chunk_i, "key_sim": key_sim}

        chunk_vec = chunks[chunk_i]

        # Unbind the KV handle: KEY_VALUE * chain(path)
        kv_handle = bind(self.roles.key_value, self.chain_vec(path))
        val_hat = unbind(chunk_vec, kv_handle)

        cleanup = self.vsa_tok.symbol_dict

        # Restrict alphabet by default to reduce false positives
        if allowed_chars is None:
            allowed_chars = (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
                " -_:/."
            )

        decoded = decode_wordvector_gb(
            val_hat,
            cleanup=cleanup,
            max_len=max_len,
            min_sim=min_sim_char,
            allowed_chars=allowed_chars,
            stop_after_misses=stop_after_misses,
        )

        return decoded, {
            "ok": True,
            "best_chunk": chunk_i,
            "key_sim": key_sim,
            "path": path,
            "decoded_len": len(decoded),
        }

    def topk_chunks_for_path(
            self,
            chunks: Sequence[VsaBase],
            path: Path,
            *,
            k: int = 3,
    ) -> List[Tuple[int, float]]:
        qk = self.vec_key_present(path)
        sims = [(i, float(hsim(qk, c))) for i, c in enumerate(chunks)]
        sims.sort(key=lambda t: t[1], reverse=True)
        return sims[:k]

    # def verify_kv_in_chunk(
    #     self,
    #     chunk_vec: VsaBase,
    #     path: Path,
    #     decoded_value: str,
    #     *,
    #     min_sim_kv: float = 0.56,
    # ) -> float:
    #     qkv = self.vec_key_value(path, decoded_value)  # RAW kv term
    #     return float(hsim(qkv, chunk_vec))

    def verify_kv_in_chunk(
            self,
            chunk_vec: VsaBase,
            path: Path,
            decoded_value: str,
    ) -> float:
        """
        Verify by regenerating the KV term (late-bound) and comparing to chunk.
        """
        qkv = self.vec_key_value(path, decoded_value)  # raw KV term (late-bound)
        return float(hsim(qkv, chunk_vec))

    # def decode_string_value_from_chunks_topk(
    #         self,
    #         chunks: Sequence[VsaBase],
    #         path: Path,
    #         *,
    #         k: int = 4,
    #         max_len: int = 64,
    #         min_sim_char: float = 0.53,
    #         stop_after_misses: int = 4,
    #         allowed_chars: Optional[Iterable[str]] = None,
    #         min_sim_kv_verify: float = 0.56,
    # ) -> Tuple[str, Dict[str, Any]]:
    #     if allowed_chars is None:
    #         allowed_chars = (
    #             "abcdefghijklmnopqrstuvwxyz"
    #             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    #             "0123456789"
    #             " -_:/."
    #         )
    #
    #     cleanup = self.vsa_tok.symbol_dict
    #     candidates = self.topk_chunks_for_path(chunks, path, k=k)
    #
    #     best = {"ok": False, "reason": "no_candidate_verified", "candidates": candidates}
    #     best_val = ""
    #
    #     for chunk_i, key_sim in candidates:
    #         chunk_vec = chunks[chunk_i]
    #
    #         kv_handle = bind(self.roles.key_value, self.chain_vec(path))
    #         val_hat = unbind(chunk_vec, kv_handle)
    #
    #         decoded = decode_wordvector_gb(
    #             val_hat,
    #             cleanup=cleanup,
    #             max_len=max_len,
    #             min_sim=min_sim_char,
    #             allowed_chars=allowed_chars,
    #             stop_after_misses=stop_after_misses,
    #         )
    #
    #         if not decoded:
    #             continue
    #
    #         kv_sim = self.verify_kv_in_chunk(
    #             chunk_vec, path, decoded, min_sim_kv=min_sim_kv_verify
    #         )
    #
    #         if kv_sim >= min_sim_kv_verify:
    #             return decoded, {
    #                 "ok": True,
    #                 "best_chunk": chunk_i,
    #                 "key_sim": key_sim,
    #                 "kv_sim": kv_sim,
    #                 "path": path,
    #                 "decoded_len": len(decoded),
    #             }
    #
    #         # keep best attempt for debugging
    #         if kv_sim > best.get("kv_sim", -1.0):
    #             best = {
    #                 "ok": False,
    #                 "reason": "kv_verify_failed",
    #                 "best_chunk": chunk_i,
    #                 "key_sim": key_sim,
    #                 "kv_sim": kv_sim,
    #                 "path": path,
    #                 "decoded": decoded,
    #             }
    #             best_val = decoded
    #
    #     return best_val, best


    def decode_string_value_from_chunks_topk(
        self,
        chunks: Sequence[VsaBase],
        path: Path,
        *,
        k: int = 4,
        max_len: int = 64,
        min_sim_char: float = 0.53,
        stop_after_misses: int = 4,
        allowed_chars: Optional[Iterable[str]] = None,
        min_sim_kv_verify: float = 0.56,
    ) -> Tuple[str, Dict[str, Any]]:

        if allowed_chars is None:
            allowed_chars = (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
                " -_:/."
            )

        cleanup = self.vsa_tok.symbol_dict
        candidates = self.topk_chunks_for_path(chunks, path, k=k)

        best_debug: Dict[str, Any] = {"ok": False, "reason": "no_candidate_verified", "candidates": candidates}
        best_val = ""

        # Build handle ONCE
        handle = bind(self.roles.key_value, self.chain_vec(path))

        for chunk_i, key_sim in candidates:
            chunk_vec = chunks[chunk_i]

            # Unbind by handle to expose shifted letters for all positions
            val_unbound = unbind(chunk_vec, handle)

            decoded = decode_wordvector_gb_from_unbound(
                val_unbound,
                cleanup=cleanup,
                max_len=max_len,
                min_sim=min_sim_char,
                allowed_chars=allowed_chars,
                stop_after_misses=stop_after_misses,
            )

            if not decoded:
                continue

            kv_sim = float(hsim(self.vec_key_value(path, decoded), chunk_vec))

            if kv_sim >= min_sim_kv_verify:
                return decoded, {
                    "ok": True,
                    "best_chunk": chunk_i,
                    "key_sim": float(key_sim),
                    "kv_sim": kv_sim,
                    "path": path,
                    "decoded_len": len(decoded),
                }

            # Keep best failed attempt for debugging
            if kv_sim > best_debug.get("kv_sim", -1.0):
                best_debug = {
                    "ok": False,
                    "reason": "kv_verify_failed",
                    "best_chunk": chunk_i,
                    "key_sim": float(key_sim),
                    "kv_sim": kv_sim,
                    "path": path,
                    "decoded": decoded,
                }
                best_val = decoded

        return best_val, best_debug

    def vec_key_value_terms(self, path: Path, value: Any) -> List[VsaBase]:
        """
        Late-bound / late-bundled KV encoding:
          For string value s of length L, emit L vectors:
            term_i = (KEY_VALUE * chain(path)) * roll(sym[s_i], i)

        For non-string values, fall back to old single-vector encoding (you can refine later).
        """
        handle = bind(self.roles.key_value, self.chain_vec(path))

        # strings: emit per-letter terms
        if isinstance(value, str):
            terms: List[VsaBase] = []
            shift = 0
            for c in value:
                shift += 1
                lv = np.roll(self.vsa_tok.symbol_dict[c], shift)
                terms.append(bind(handle, lv))
            return terms

        # everything else: keep old behaviour (single term based on string form)
        vv = self.value_vectoriser(str(value))
        return [bind(handle, vv)]