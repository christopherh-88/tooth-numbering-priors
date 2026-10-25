"""Power of the GAP_RANK_RULES.md thresholds (simulation, no real ranking data).

    python gap_rank_power.py

Flag size and catch rate are Section 84, seed 0, YOLOv8x. Teeth are treated as
independent, which makes the pass rates optimistic. Seed 1 of numpy default_rng.
"""
import numpy as np
from scipy.stats import norm, rankdata

rng = np.random.default_rng(1)
# Section 84, seed 0, YOLOv8x: 1327 flagged of ~11,420 matched teeth, 73 of 161 joint failures flagged
NF, JF, JT, NM = 1327, 73, 161, 11420


def auc(s, y):
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def one(a, nB, R, nb=120):
    d = np.sqrt(2) * norm.ppf(a)
    y = np.zeros(NF, int)
    y[:JF] = 1
    s = rng.normal(size=NF) + d * y
    top = np.argsort(-s)[:nB]
    caught = y[top].sum()
    trunc = JF * nB / NF
    diffs, aus = [], []
    for _ in range(nb):
        ii = rng.integers(0, NF, NF)
        yb, sb = y[ii], s[ii]
        t = np.argpartition(-sb, nB)[:nB]
        diffs.append(yb[t].sum() - yb.sum() * nB / NF)
        aus.append(auc(sb, yb))
    rec = caught / trunc >= R and np.percentile(diffs, 2.5) > 0
    aur = auc(s, y) >= 0.65 and np.percentile(aus, 2.5) > 0.5
    return rec, aur


print("budget%  nB  trunc_recall%  max_recall%")
for B in (2, 4, 6, 8):
    nB = int(round(B / 100 * NM))
    print(B, nB, round(100 * JF * nB / NF / JT, 1), round(100 * min(JF, nB) / JT, 1))

NSIM = 100
print("\npass rate per detector (teeth treated as independent, so optimistic)")
for a in (0.5, 0.6, 0.7, 0.8):
    row = []
    for B, R in ((8, 1.25), (4, 1.5), (2, 1.5)):
        nB = int(round(B / 100 * NM))
        res = [one(a, nB, R) for _ in range(NSIM)]
        row.append(f"B{B}/R{R}: {np.mean([r[0] for r in res]):.2f}")
        if B == 4:
            row.append(f"AUROC rule: {np.mean([r[1] for r in res]):.2f}")
    print("true AUROC", a, " | ".join(row), flush=True)
