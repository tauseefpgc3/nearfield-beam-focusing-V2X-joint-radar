"""D-036 Experiment A: grouped 5-fold sequence cross-validation inputs for DeepSense 32-34.

    python scripts/phase4/a_prepare.py 32 33 34

Folds: unique sequences shuffled with default_rng(20261118) (Phase 3c split seed), cut into 5 at
cumulative frame fractions 0.2/0.4/0.6/0.8 (the Phase 3c cumulative-fraction rule). In fold k the
other folds' sequences are shuffled with default_rng(20261118 + k); validation = leading
sequences up to 25% of their frames (same rule), the rest training. Grid design (ds_prepare.design,
frozen) from every 5th training frame of the fold; BEVs with ds_prepare.bev. Writes
data/processed/phase4/cv/f{k}/ds{S}.h5 (same layout as Phase 3c) and results/phase4/a_folds_ds{S}.json.
"""
import json, os, sys
from multiprocessing import Pool
import h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3c")); import ds_prepare as DP
OUT = os.path.join(ROOT, "data", "processed", "phase4", "cv"); RES = os.path.join(ROOT, "results", "phase4")


def cut(seqs, cnt, n, edges, rng):
    u = np.array(seqs); rng.shuffle(u); out, cum = {}, 0
    for q in u:
        out[int(q)] = int(np.searchsorted(edges, cum / n, side="right")); cum += cnt[q]
    return out


def folds(seq):
    cnt = {q: int((seq == q).sum()) for q in np.unique(seq)}
    fold = cut(np.unique(seq), cnt, len(seq), [0.2, 0.4, 0.6, 0.8], np.random.default_rng(DP.SPLIT_SEED)); sp = {}
    for k in range(5):
        pool = [q for q in np.unique(seq) if fold[int(q)] != k]; npool = sum(cnt[q] for q in pool)
        role = cut(pool, cnt, npool, [0.25], np.random.default_rng(DP.SPLIT_SEED + k))   # 0 -> val, 1 -> train
        s = np.full(len(seq), 2, np.int8)
        for q in pool: s[seq == q] = 1 if role[int(q)] == 0 else 0
        sp[k] = s
    return fold, sp


def main(s):
    df, lid, pw = DP.index(s); n = len(df); seq = df.seq_index.to_numpy(); fold, sp = folds(seq)
    with Pool(8) as pool:
        P = np.stack(pool.map(DP.load_power, pw, chunksize=64)); clouds = pool.map(DP.load_points, [(s, p) for p in lid], chunksize=16)
    info = dict(scenario=s, frames=n, sequences=int(len(np.unique(seq))), fold_of_sequence=fold, folds={})
    for k in range(5):
        d = os.path.join(OUT, f"f{k}"); os.makedirs(d, exist_ok=True); split = sp[k]; tr = np.where(split == 0)[0]
        g = DP.design([clouds[i] for i in tr[::5]], False)
        with h5py.File(os.path.join(d, f"ds{s}.h5"), "w") as hf:
            hf.create_dataset("bev", data=np.stack([DP.bev(c, g) for c in clouds]), chunks=(1, 3, g["NX"], g["NY"]))
            hf["power"] = P.astype(np.float32); hf["label"] = P.argmax(1).astype(np.int16); hf["seq"] = seq; hf["split"] = split
            hf["row"] = np.arange(n); hf["n_points"] = np.array([len(c) for c in clouds], np.int32)
            for kk, v in g.items(): hf.attrs[kk] = v
        info["folds"][k] = dict(grid=g, frames={nm: int((split == i).sum()) for i, nm in enumerate(("train", "val", "test"))},
                                sequences={nm: int(len(np.unique(seq[split == i]))) for i, nm in enumerate(("train", "val", "test"))})
        print(s, k, info["folds"][k], flush=True)
    os.makedirs(RES, exist_ok=True); json.dump(info, open(os.path.join(RES, f"a_folds_ds{s}.json"), "w"), indent=1)


if __name__ == "__main__":
    for a in sys.argv[1:]: main(int(a))
