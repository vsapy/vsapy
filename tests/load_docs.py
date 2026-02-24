import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import matplotlib.pyplot as plt

from vsapy import hsim
from vsapy.bag import BareChunk
from vsapy.helpers import deserialise_object


# -----------------------------------------------
# Load docs (build the docs using build_docs.py)
# -----------------------------------------------
def load_vsa_document_set(usechunksforWords,
                          use_word2vec,
                          use_shaping,
                          no_acts=100, no_scenes_per_act=100,
                          prepend_level_labels=False):

    word_format = ''
    word_format += '_CW-T' if usechunksforWords else 'CW-F'
    word_format += '_WV-T' if use_word2vec else '_WV-F'
    word_format += '_PVECS' if usechunksforWords else '_CHAIN'
    word_format += '_SHAPED' if use_shaping else '_NoShape'

    file_path = './data/output_vec_files'

    runlist = [
        ('macbeth', f'{file_path}/macbeth_nltk_{word_format}.bin', 'Macbeth', 'OE_tk_mbeth',
         'Macbeth: Old English. Natural Language Tool Kit.'),
        ('ntlk_shakespear', f'{file_path}/hamlet_nltk_{word_format}.bin', 'Hamlet', 'OE_tk_ham',
         'Hamlet: Old English. Natural Language Tool Kit.'),
        ('hamlet_orig', f'{file_path}/hamlet_stanzas_{word_format}.bin', 'Hamlet', 'OE_og_ham',
         'Hamlet: Old English. Original Source.'),
        ('nofear_old', f'{file_path}/hamlet_old_{word_format}.bin', 'Hamlet', 'OE_nf_ham',
         'Hamlet: Old English. NoFear Translation Source.'),
        ('nofear_new', f'{file_path}/hamlet_new_{word_format}.bin', 'Hamlet', 'NE_nf_ham',
         'Hamlet: New English. NoFear Translation Source.')
    ]

    docs = {}
    for kk, fn, play_name, short_info, long_info in runlist:
        top_chunk = deserialise_object(fn, None)
        if prepend_level_labels:
            BareChunk.add_level_labels(top_chunk, 0, short_info)
        docs[short_info] = top_chunk

    return docs


# ----------------------------
# Parse hierarchical chunk names
# ----------------------------
@dataclass(frozen=True)
class ChunkAddress:
    doc_key: str               # e.g. OE_tk_ham
    indices: Tuple[int, ...]   # e.g. (0,5,2) for $00-05-02
    title: str                 # e.g. "ACT V_SCENE II..."

    @property
    def depth(self) -> int:
        # 1 => DOC ($00), 2 => ACT ($00-05), 3 => SCENE ($00-05-02)
        return len(self.indices)

    @property
    def act_index(self) -> Optional[int]:
        return self.indices[1] if self.depth >= 2 else None

    @property
    def scene_index(self) -> Optional[int]:
        return self.indices[2] if self.depth >= 3 else None


_chunk_re = re.compile(r"^(?P<doc>[^$]+)\$(?P<pos>[^@]+)@(?P<title>.*)$")


def parse_chunk_name(aname: str) -> ChunkAddress:
    """
    Examples:
      OE_tk_ham$00-05@ACT V
      OE_tk_ham$00-05-02@ACT V_SCENE II. A hall...
    """
    m = _chunk_re.match(aname)
    if not m:
        # fallback,
        return ChunkAddress(doc_key=aname, indices=(0,), title="(unparsed)")

    doc_key = m.group("doc")
    pos = m.group("pos")
    title = m.group("title").strip()

    parts = pos.split("-")
    indices = tuple(int(p) for p in parts)  # handles leading zeros fine (e.g., "05" -> 5)
    return ChunkAddress(doc_key=doc_key, indices=indices, title=title)


def group_name_from_addr(addr: ChunkAddress) -> Optional[str]:
    """
    We only group DOC and ACT for now:
      DOC: indices=(0,)
      ACT: indices=(0, act)
    """
    if addr.depth == 1:
        return "DOC"
    if addr.depth == 2:
        return f"ACT {addr.act_index:02d}"
    return None  # ignore scenes in this demo


# ----------------------------
# Pair labels + canonicalisation
# ----------------------------
def short_doc_key(k: str) -> str:
    # Keep it readable but shorter than raw underscores
    return k.replace("_", "-")


def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    # Canonical ordering so A↔B and B↔A collapse to same key
    aa, bb = sorted([a, b])
    return aa, bb


def pair_label(a: str, b: str) -> str:
    aa, bb = canonical_pair(a, b)
    return f"{aa}↔{bb}"


def is_macbeth_pair(pair_lbl: str) -> bool:
    return "mbeth" in pair_lbl


def is_hamlet_pair(pair_lbl: str) -> bool:
    # Hamlet variants include "-ham" and do not include "mbeth"
    return ("-ham" in pair_lbl) and not is_macbeth_pair(pair_lbl)


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def bucket(pair_lbl: str, score: float, threshold: float) -> int:
    """
    0 = Pass Hamlet (left)
    1 = Fail Hamlet (middle)
    2 = Macbeth comparisons (right)
    """
    if is_macbeth_pair(pair_lbl):
        return 2
    # Treat all non-macbeth pairs as "hamlet family" for ordering purposes
    return 0 if score >= threshold else 1


# ----------------------------
# Plotting
# ----------------------------
def plot_group(group: str,
               values_by_pair: Dict[str, List[float]],
               threshold: float,
               y_min: float,
               y_max: float):
    if not values_by_pair:
        print(f"[plot] No data for {group}")
        return

    scored = [(p, mean(vals)) for p, vals in values_by_pair.items()]
    scored.sort(key=lambda t: (bucket(t[0], t[1], threshold), -t[1], t[0]))

    pairs = [p for p, _ in scored]
    xs = list(range(len(pairs)))

    plt.figure(figsize=(max(11, len(pairs) * 1.1), 5))

    # scatter points
    for i, p in enumerate(pairs):
        ys = values_by_pair[p]
        if len(ys) == 1:
            plt.scatter([i], ys, label=p, s=70)
        else:
            # If we add multiple pairs, jitter them to make them viewable.
            jitter = [i + (j - (len(ys) - 1) / 2) * 0.08 for j in range(len(ys))]
            plt.scatter(jitter, ys, label=p, s=40)

    # Horizontal match threshold,
    # ToDo: we could get this dynamically
    plt.axhline(threshold, linewidth=2)

    # separators between Pass Hamlet | Fail Hamlet | Macbeth
    buckets = [bucket(p, mean(values_by_pair[p]), threshold) for p in pairs]
    for i in range(1, len(buckets)):
        if buckets[i] != buckets[i - 1]:
            plt.axvline(i - 0.5, linewidth=1)

    plt.xticks(xs, pairs, rotation=35, ha="right")
    plt.ylabel("hsim")
    plt.title(f"{group}: Pass Hamlet | Fail Hamlet | Macbeth")
    plt.ylim(y_min, y_max)

    # Legend outside the plot area
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout()


# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    # Default BSC threshold
    THRESHOLD = 0.525

    # Which levels to extract:
    #   [0,1] gives DOC + ACT vectors
    levels_to_extract = [0, 1]

    docs = load_vsa_document_set(
        usechunksforWords=True,
        use_word2vec=False,
        use_shaping=False,
        prepend_level_labels=True
    )

    # Extract chunks for each document
    doc_chunks: Dict[str, List[BareChunk]] = {}
    for k, top in docs.items():
        chunks: List[BareChunk] = []
        BareChunk.get_levels_as_list(top, levels_to_extract, 0, chunks)
        doc_chunks[k] = chunks

    # Grouped collection:
    #   grouped[group_name][pair_label] -> list[hsim]
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    # Compare like-with-like via hierarchical indices:
    # DOC: (0,) compares with (0,)
    # ACT i: (0,i) compares with (0,i)
    for k1, chunks1 in doc_chunks.items():
        for k2, chunks2 in doc_chunks.items():
            if k1 == k2:
                continue

            k1s = short_doc_key(k1)
            k2s = short_doc_key(k2)
            p_lbl = pair_label(k1s, k2s)  # canonicalised here

            for c1 in chunks1:
                a1 = parse_chunk_name(c1.aname)
                g1 = group_name_from_addr(a1)
                if g1 is None:
                    continue

                for c2 in chunks2:
                    a2 = parse_chunk_name(c2.aname)
                    g2 = group_name_from_addr(a2)
                    if g2 is None:
                        continue

                    # Must compare the same hierarchical position (same indices)
                    if a1.indices != a2.indices:
                        continue

                    s = float(hsim(c1.myvec, c2.myvec))
                    if grouped[g1][p_lbl]:
                        if s in grouped[g1][p_lbl]:
                            continue
                    grouped[g1][p_lbl].append(s)

    # Plot DOC
    plot_group("DOC", grouped.get("DOC", {}), threshold=THRESHOLD, y_min=0.48, y_max=0.65)

    # Plot ACTs in order
    act_groups = sorted([g for g in grouped.keys() if g.startswith("ACT ")])
    for g in act_groups:
        plot_group(g, grouped[g], threshold=THRESHOLD, y_min=0.48, y_max=0.75)

    plt.show()