import numpy as np

from vsapy.vsapy import cosine_sim, hsim, randvec, hdist
from vsapy.vsatype import VsaType
from vsapy.bag import BagVec, ShiftedBag
from vsapy.laiho import Laiho

ALPHABET = list("abcdefghijklmnopqrstuvwxyz")


def make_randvec(vt: VsaType, vd: int, *, bits_per_slot: int = 32):
    if Laiho.is_laiho_type(vt):
        slots = Laiho.slots_from_bsc_vec(vd, bits_per_slot)
        return randvec(slots, vsa_type=vt, bits_per_slot=bits_per_slot)
    return randvec(vd, vsa_type=vt)


def build_alphabet_bank(vt: VsaType, vd: int, *, bits_per_slot: int = 32):
    vecs = [make_randvec(vt, vd, bits_per_slot=bits_per_slot) for _ in ALPHABET]
    bank = np.stack(vecs, axis=0)  # (26, D) => banked ops go fast
    return bank, {c: vecs[i] for i, c in enumerate(ALPHABET)}


def word_vec_shifted(word: str, token_map: dict[str, np.ndarray]):
    # ShiftedBag rolls each subvec by 1..N before bundling => positional sensitivity
    vecs = [token_map[c] for c in word if c in token_map]
    if not vecs:
        raise ValueError(f"No valid tokens in {word!r}")
    return ShiftedBag(vecs).myvec


def noisify(v, vt: VsaType, rng: np.random.Generator, p: float = 0.05):
    vv = v.copy()
    if vt == VsaType.BSC:
        mask = rng.random(vv.shape) < p
        vv[mask] = 1 - vv[mask]
        return vv

    if vt in (VsaType.Tern, VsaType.TernZero):
        # vsapy convention: randvec gives {-1,+1}. Zeros may only appear from bundling ties (TernZero).
        vv = np.asarray(vv, dtype=np.int8).copy()
        mask = np.asarray(rng.random(vv.shape) < p)

        nonzero = (vv != 0)
        flip_mask = mask & nonzero
        vv[flip_mask] *= -1

        # For TernZero: keep zeros as zeros (they encode ties), don't "fill them in"
        return vv

    if vt == VsaType.HRR:
        vv = vv.astype(np.float32, copy=True)
        vv += 0.05 * rng.standard_normal(vv.shape).astype(np.float32)
        vv /= (np.linalg.norm(vv) + 1e-12)
        return vv

    return vv


def report_retrieval(name_list, scores, *, similarity: bool, topk: int = 5, label: str = ""):
    scores = np.asarray(scores)

    if similarity:
        best_i = int(np.argmax(scores))      # <-- explicit argmax
        order = np.argsort(scores)[::-1][:topk]
        op = "argmax"
    else:
        best_i = int(np.argmin(scores))      # <-- explicit argmin
        order = np.argsort(scores)[:topk]
        op = "argmin"

    best_name = name_list[best_i]
    best_score = float(scores[best_i])

    hdr = f"    {label} {op}(scores) -> index={best_i}, item={best_name!r}, score={best_score:0.4f}".rstrip()
    print(hdr)
    print("    topk indices:", list(map(int, order)))
    print("    topk items  :", ", ".join([f"{name_list[i]}({float(scores[i]):0.4f})" for i in order]))


def run_banked_shifted_demo(
    vd: int = 4096,
    bits_per_slot: int = 32,
    vsa_types=(VsaType.BSC, VsaType.Tern, VsaType.TernZero, VsaType.HRR, VsaType.Laiho),
    words=("cat", "tac", "cap", "cape", "dog", "dogs", "dot", "data", "date", "mate", "math", "hamlet", "macbeth"),
    query_letter="t",
    query_word="hamlet",
    topk: int = 5,
    seed: int = 1234,
):
    rng = np.random.default_rng(seed)

    for vt in vsa_types:
        print("\n" + "=" * 90)
        print(f"VSA TYPE: {vt.name}")

        # ----------------------------
        # 1) Alphabet bank retrieval
        # ----------------------------
        alpha_bank, alpha_map = build_alphabet_bank(vt, vd, bits_per_slot=bits_per_slot)
        q_letter = noisify(alpha_map[query_letter], vt, rng, p=0.05)

        print(f"\n  Alphabet-bank query: {query_letter!r} (noisy)")

        if vt != VsaType.HRR:
            s = hsim(q_letter, alpha_bank)
            print("  hsim:")
            report_retrieval(ALPHABET, s, similarity=True, topk=topk)

            d = hdist(q_letter, alpha_bank)
            print("  hdist:")
            report_retrieval(ALPHABET, d, similarity=False, topk=topk)

        # cosine on all types; for BSC it may be more meaningful in bipolar,...
        try:
            cs = cosine_sim(q_letter, alpha_bank)
        except Exception:
            if vt == VsaType.BSC:
                q = q_letter.astype(np.float32) * 2.0 - 1.0
                B = alpha_bank.astype(np.float32) * 2.0 - 1.0
                cs = cosine_sim(q, B)
            else:
                cs = None
        if cs is not None:
            print("  cosine_sim:")
            report_retrieval(ALPHABET, cs, similarity=True, topk=topk)

        # ----------------------------
        # 2) Word bank retrieval using ShiftedBag
        # ----------------------------
        word_vecs = [word_vec_shifted(w, alpha_map) for w in words]
        word_bank = np.stack(word_vecs, axis=0)

        q_word = word_vec_shifted(query_word, alpha_map)
        q_word = noisify(q_word, vt, rng, p=0.05)

        print(f"\n  Word-bank query (ShiftedBag positional): {query_word!r} (noisy)")
        if vt != VsaType.HRR:
            s = hsim(q_word, word_bank)
            print("  hsim:")
            report_retrieval(words, s, similarity=True, topk=topk)

            d = hdist(q_word, word_bank)
            print("  hdist:")
            report_retrieval(words, d, similarity=False, topk=topk)

        try:
            cs = cosine_sim(q_word, word_bank)
        except Exception:
            if vt == VsaType.BSC:
                q = q_word.astype(np.float32) * 2.0 - 1.0
                B = word_bank.astype(np.float32) * 2.0 - 1.0
                cs = cosine_sim(q, B)
            else:
                cs = None
        if cs is not None:
            print("  cosine_sim:")
            report_retrieval(words, cs, similarity=True, topk=topk)

        if "cat" in words and "tac" in words:
            i_cat = words.index("cat")
            i_tac = words.index("tac")
            try:
                cat_tac = hsim(word_bank[i_cat], word_bank[i_tac]) if vt != VsaType.HRR else cosine_sim(word_bank[i_cat], word_bank[i_tac])
                print(f"\n  Sanity: similarity('cat','tac') with ShiftedBag = {float(cat_tac):0.4f}")
            except Exception:
                pass


if __name__ == "__main__":
    run_banked_shifted_demo()
