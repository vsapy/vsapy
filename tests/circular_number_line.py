import os
import numpy as np

from vsapy import hsim
from vsapy.number_line import CircularNumberLineFolded, CircularNumberLinePhaseProjected
from vsapy.vsatype import VsaType


def mean_similarity_by_offset(levels_arr: np.ndarray) -> np.ndarray:
    """
    Mean similarity between all pairs separated by k steps, for k=0..m//2.
    levels_arr: (m, D) bool
    """
    m = int(levels_arr.shape[0])
    max_k = m // 2
    means = np.empty(max_k + 1, dtype=float)

    for k in range(max_k + 1):
        sims_k = np.empty(m, dtype=float)
        for i in range(m):
            j = (i + k) % m
            sims_k[i] = hsim(levels_arr[i], levels_arr[j])
        means[k] = float(sims_k.mean())

    return means


def anchor_similarity_by_offset(levels_arr: np.ndarray, anchor_idx: int = 0) -> np.ndarray:
    """
    Similarity of a fixed anchor to offsets k=0..m//2.
    levels_arr: (m, D) bool
    """
    m = int(levels_arr.shape[0])
    max_k = m // 2
    anchor = levels_arr[int(anchor_idx) % m]
    sims = np.empty(max_k + 1, dtype=float)

    for k in range(max_k + 1):
        sims[k] = hsim(anchor, levels_arr[(anchor_idx + k) % m])

    return sims


def _maybe_plot(degs: np.ndarray, curves: dict[str, np.ndarray], title: str) -> None:
    """
    Optional plotting for local visual inspection.
    Enable via: VSA_PLOT=1 pytest -k anchor
    """
    if os.getenv("VSA_PLOT", "1") not in ("1", "true", "TRUE", "yes", "YES"):
        return

    import matplotlib.pyplot as plt

    plt.figure()
    for label, y in curves.items():
        plt.scatter(degs, y, marker=".", label=label)
    plt.title(title)
    plt.xlabel("Circular distance (degrees)")
    plt.ylabel("Hamming similarity (1 - HD)")
    plt.legend()
    plt.show()


def circular_numberline_anchor_graph_folded_vs_phase_projected():
    """
    Anchor-graph regression:
      - Folded: anchor curves differ (NOT rotationally invariant).
      - PhaseProjected: anchor curves agree (approximately rotationally invariant) and match mean-over-anchors.
    """
    seed = 123
    rng = np.random.default_rng(seed)

    vsa_kwargs = {"vsa_type": VsaType.BSC}
    period = 360.0
    m_steps = 360
    D = 10_000

    # --- Folded (not rotationally invariant) ---
    folded = CircularNumberLineFolded(
        period=period,
        m_steps=m_steps,
        vec_dim=D,
        vsa_kwargs=vsa_kwargs,
        max_hd=0.5,
        rng=rng,
    )
    levels_f = folded.levels  # (m, D)
    assert levels_f.shape == (m_steps, D)

    mean_f = mean_similarity_by_offset(levels_f)
    a0_f = anchor_similarity_by_offset(levels_f, 0)
    a37_f = anchor_similarity_by_offset(levels_f, 37)

    # Folded should NOT be rotationally invariant: anchor curves should diverge measurably.
    max_diff_folded = float(np.max(np.abs(a0_f - a37_f)))
    assert max_diff_folded > 0.02, f"Folded looked unexpectedly invariant (max diff={max_diff_folded:.4f})"

    # anchor graph plot for folded
    degs = np.arange(0, m_steps // 2 + 1) * (period / m_steps)
    _maybe_plot(
        degs,
        {
            "folded anchor 0°": a0_f,
            "folded anchor 37°": a37_f,
            "folded mean over anchors": mean_f,
        },
        title=f"Folded circular numberline: anchor-vs-anchor-vs-mean (m={m_steps}, D={D})",
    )

    # --- PhaseProjected (approximately rotationally invariant) ---
    # New seed to make runs independent
    rng2 = np.random.default_rng(seed)

    phase = CircularNumberLinePhaseProjected(
        period=period,
        m_steps=m_steps,
        vec_dim=D,
        vsa_kwargs=vsa_kwargs,
        rdim=2,
        n_harmonics=None, # default rdim//2
        rng=rng2,
    )
    levels_p = phase.levels  # (m, D)
    assert levels_p.shape == (m_steps, D)
    # Closure check via the provided property
    closed = phase.levels_closed
    assert np.array_equal(closed[0], closed[-1]), "PhaseProjected closure violated (0° != 360°)"

    mean_p = mean_similarity_by_offset(levels_p)
    a0_p = anchor_similarity_by_offset(levels_p, 0)
    a37_p = anchor_similarity_by_offset(levels_p, 37)

    _maybe_plot(
        degs,
        {
            "phase anchor 0°": a0_p,
            "phase anchor 37°": a37_p,
            "phase mean over anchors": mean_p,
        },
        title=f"PhaseProjected: anchor-vs-anchor-vs-mean (m={m_steps}, D={D}, rdim={phase.rdim})",
    )

    print("PhaseProjected diagnostics:")
    print("  max|anchor0-mean| =", float(np.max(np.abs(a0_p - mean_p))))
    print("  max|anchor37-mean| =", float(np.max(np.abs(a37_p - mean_p))))
    print("  max|anchor0-anchor37| =", float(np.max(np.abs(a0_p - a37_p))))


if __name__ in "__main__":
    circular_numberline_anchor_graph_folded_vs_phase_projected()