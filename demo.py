#!/usr/bin/env python3
# Self-contained demo: segment a DNA sequence into CpG-island vs background
# regions with a Hidden Markov Model. The island state emits G/C-rich sequence;
# Viterbi recovers the most-likely segmentation and forward-backward gives the
# per-base posterior probability of being in an island. All from scratch.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hmm import viterbi, forward_backward, simulate

BASES = np.array(list("ACGT"))
# State 0 = background (AT-rich, GC~0.40); state 1 = CpG island (GC-rich, GC~0.70).
INIT = np.array([0.95, 0.05])
TRANS = np.array([[0.9990, 0.0010], [0.005, 0.995]])
EMIT = np.array([[0.30, 0.20, 0.20, 0.30], [0.15, 0.35, 0.35, 0.15]])
N = 6000


def segments(mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None: start = i
        elif not v and start is not None: out.append((start, i)); start = None
    if start is not None: out.append((start, len(mask)))
    return out


def main():
    os.makedirs("figures", exist_ok=True); os.makedirs("results", exist_ok=True)
    states, obs = simulate(INIT, TRANS, EMIT, N, seed=2)
    path = viterbi(obs, INIT, TRANS, EMIT)
    gamma = forward_backward(obs, INIT, TRANS, EMIT)
    post_island = gamma[:, 1]

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    # Panel 1: GC content track with true vs predicted islands.
    a = ax[0, 0]
    isgc = np.isin(obs, [1, 2]).astype(float)                 # C or G
    w = 60; gc = np.convolve(isgc, np.ones(w)/w, mode="same")
    a.plot(gc, color="#333333", lw=0.6)
    a.axhline(0.5, color="grey", lw=0.5)
    for s, e in segments(states == 1):
        a.axvspan(s, e, color="#55A868", alpha=0.25)
    for s, e in segments(path == 1):
        a.axvspan(s, e, facecolor="none", edgecolor="#C44E52", lw=1.2, hatch="//")
    a.set_xlabel("position (bp)"); a.set_ylabel("GC content (100 bp)")
    a.set_title("GC track: true islands (green) vs Viterbi (red hatch)")

    # Panel 2: posterior probability of the island state.
    a = ax[0, 1]
    a.plot(post_island, color="#4C72B0", lw=0.7)
    for s, e in segments(states == 1):
        a.axvspan(s, e, color="#55A868", alpha=0.2)
    a.set_ylim(-0.02, 1.02); a.set_xlabel("position (bp)"); a.set_ylabel("P(island)")
    a.set_title("Forward-backward posterior (green = true islands)")

    # Panel 3: emission probabilities per state.
    a = ax[1, 0]
    x = np.arange(4)
    a.bar(x - 0.2, EMIT[0], 0.4, label="background", color="#DD8452")
    a.bar(x + 0.2, EMIT[1], 0.4, label="island", color="#55A868")
    a.set_xticks(x); a.set_xticklabels(BASES); a.set_ylabel("emission probability")
    a.set_title("The island state is G/C-rich"); a.legend(fontsize=8)

    # Panel 4: recovery accuracy and island counts.
    a = ax[1, 1]
    acc = float((path == states).mean())
    n_true = len(segments(states == 1)); n_pred = len(segments(path == 1))
    bars = a.bar(["Viterbi\naccuracy", "true\nislands", "predicted\nislands"],
                 [acc, n_true, n_pred], color=["#4C72B0", "#55A868", "#C44E52"])
    a.set_title("Segmentation recovery")
    for b, v in zip(bars, [acc, n_true, n_pred]):
        a.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}" if v < 1.5 else str(int(v)), ha="center", fontsize=9)

    fig.suptitle("CpG-island segmentation with an HMM (synthetic sequence)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig("figures/demo.png", dpi=120)

    pd.DataFrame([{"length_bp": N, "viterbi_accuracy": acc,
                   "true_islands": n_true, "predicted_islands": n_pred,
                   "bases_in_true_islands": int((states == 1).sum())}]).to_csv("results/summary.csv", index=False)
    print(f"Viterbi accuracy: {acc:.3f}  true islands: {n_true}  predicted: {n_pred}")
    print("Wrote figures/demo.png and results/summary.csv")


if __name__ == "__main__":
    main()
