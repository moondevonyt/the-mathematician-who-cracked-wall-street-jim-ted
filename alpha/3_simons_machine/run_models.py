"""
Moon Dev: run every walk-forward config and cache the out-of-sample scores (cache/scores_*.npy).
usage: python run_models.py A|B|C|D   (groups so they run side by side)
config = (bar hours, model, feature set, retrain, target hours: 0 = next bar)
"""
import sys, pickle, time
import numpy as np
import machine as M

M4 = ["trend20", "best1", "blend", "ridge", "gbm"]
GROUPS = {
    "A": [(4, m, "all", "monthly", 0) for m in M4] + [(4, m, "all", r, 0) for m in ["blend", "ridge", "gbm"] for r in ["quarterly", "once"]],
    "B": [(4, m, fs, "monthly", 0) for m in ["blend", "ridge", "gbm"] for fs in ["price", "price+volume", "price+volume+flow"]],
    "C": [(1, m, "all", "monthly", 0) for m in M4] + [(1, m, "all", "once", 0) for m in ["ridge", "gbm"]],
    "D": [(4, m, "all", "monthly", 24) for m in ["best1", "blend", "ridge"]]
         + [(4, m, "all", r, 24) for m in ["blend", "ridge"] for r in ["quarterly", "once"]]
         + [(4, m, fs, "monthly", 24) for m in ["blend", "ridge"] for fs in ["price", "price+volume", "price+volume+flow"]]
         + [(4, "gbm", "all", "once", 24), (4, "gbm", "all", "monthly", 24)],
}


def key(h, m, fs, r, tgt=0):
    return f"{h}h_{m}_{fs.replace('+', '-')}_{r}" + (f"_t{tgt}h" if tgt else "")


if __name__ == "__main__":
    g = sys.argv[1]
    P = {}
    for h, m, fs, r, tgt in GROUPS[g]:
        k = key(h, m, fs, r, tgt)
        if (M.CACHE / f"scores_{k}.npy").exists():
            continue
        if (h, tgt) not in P:
            P[(h, tgt)] = M.panel(h, tgt)
        t0 = time.time()
        S, infos = M.walk_forward(P[(h, tgt)], m, M.FEATSETS[fs], r)
        np.save(M.CACHE / f"scores_{k}.npy", S)
        pickle.dump(infos, open(M.CACHE / f"infos_{k}.pkl", "wb"))
        print(f"Moon Dev: {k} done in {time.time() - t0:.0f}s", flush=True)
