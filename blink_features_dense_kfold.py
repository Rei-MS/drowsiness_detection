"""
Paso 2 — modelo sobre features de parpadeo REALES (EAR denso 15fps).

Ventanas de ~30s sobre la secuencia de EAR de cada video → features de parpadeo
(PERCLOS, blink rate, duración, microsueños, stats de EAR) → HistGradientBoosting
con 5-fold CV subject-independent + agregación por video.

Uso:
    python blink_features_dense_kfold.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

NPZ       = "data/processed/ear_dense.npz"
KEEP      = [0, 2]
LABEL_MAP = {0: 0, 2: 1}
K         = 5
FPS       = 15
WIN       = 450      # 30 s
STRIDE    = 150      # 10 s
THR       = 0.21
THR2      = 0.25
SEED      = 42


def window_features(e: np.ndarray) -> np.ndarray:
    closed = e < THR
    d = np.abs(np.diff(e)) if len(e) > 1 else np.array([0.0])

    # cierres (parpadeos / microsueños)
    durs, i = [], 0
    while i < len(closed):
        if closed[i]:
            j = i
            while j < len(closed) and closed[j]:
                j += 1
            durs.append((j - i) / FPS)
            i = j
        else:
            i += 1
    durs = np.array(durs) if durs else np.array([0.0])
    n_blinks = int(np.sum((~closed[:-1]) & (closed[1:]))) if len(e) > 1 else 0
    secs = len(e) / FPS

    return np.array([
        e.mean(), e.std(), e.min(), np.median(e),
        np.percentile(e, 10), np.percentile(e, 25),
        closed.mean(),                      # PERCLOS
        (e < THR2).mean(),                  # PERCLOS laxo
        n_blinks / (secs / 60.0),           # parpadeos por minuto
        durs.mean(), durs.max(), durs.std(),# duración de cierres
        float(np.sum(durs > 0.5)),          # microsueños (>0.5s)
        float(np.sum(durs > 1.0)),          # cierres largos (>1s)
        durs.max(),                         # cierre más largo
        d.mean(), d.std(),                  # velocidad de EAR
    ], dtype=np.float32)


def build():
    d = np.load(NPZ, allow_pickle=True)
    ear, subj, lbl, vid = d["ear"], d["subject"], d["label"], d["video_id"]
    X, y, S, V = [], [], [], []
    for v in np.unique(vid):
        m = (vid == v)
        cls = int(lbl[m][0])
        if cls not in KEEP:
            continue
        seq = ear[m]
        if len(seq) < WIN:
            continue
        s = subj[m][0]
        for st in range(0, len(seq) - WIN + 1, STRIDE):
            X.append(window_features(seq[st:st + WIN]))
            y.append(LABEL_MAP[cls]); S.append(s); V.append(v)
    return np.array(X), np.array(y), np.array(S), np.array(V)


def main():
    X, y, S, V = build()
    print(f"Ventanas: {len(X):,}  features: {X.shape[1]}  "
          f"clases: {dict(zip(*np.unique(y, return_counts=True)))}")

    rng = np.random.default_rng(SEED)
    su = np.unique(S); rng.shuffle(su)
    folds = np.array_split(su, K)
    res = []

    for i in range(K):
        test_s = set(folds[i].tolist())
        tr = np.array([s not in test_s for s in S]); te = ~tr
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, random_state=SEED)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        win_f1 = f1_score(y[te], (proba >= 0.5).astype(int), average="macro", zero_division=0)

        vids = V[te]; yt = y[te]; vt, vp, vs = [], [], []
        for v in np.unique(vids):
            vm = vids == v
            vt.append(int(yt[vm][0])); vp.append(int(proba[vm].mean() >= 0.5))
            vs.append(proba[vm].mean())
        vt, vp = np.array(vt), np.array(vp)
        acc = accuracy_score(vt, vp); vf1 = f1_score(vt, vp, average="macro", zero_division=0)
        auc = roc_auc_score(vt, vs) if len(set(vt)) > 1 else float("nan")
        print(f">> FOLD {i+1}: win_F1={win_f1:.4f}  video_acc={acc:.4f}  "
              f"video_F1={vf1:.4f}  video_AUC={auc:.4f}  (n={len(vt)})")
        res.append({"fold": i, "win_f1": float(win_f1), "video_accuracy": float(acc),
                    "video_f1": float(vf1), "video_auc": float(auc), "n_videos": len(vt)})

    va = np.array([r["video_accuracy"] for r in res])
    vf = np.array([r["video_f1"] for r in res])
    au = np.array([r["video_auc"] for r in res])
    summary = {"model": "blink_dense_histgbm", "k": K, "window_s": WIN / FPS,
               "folds": res,
               "video_acc_mean": float(va.mean()), "video_acc_std": float(va.std()),
               "video_f1_mean": float(vf.mean()), "video_f1_std": float(vf.std()),
               "video_auc_mean": float(np.nanmean(au))}
    Path("checkpoints/kfold").mkdir(parents=True, exist_ok=True)
    with open("checkpoints/kfold/summary_blink_dense.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESUMEN blink-dense (15fps) — {K}-fold CV subject-independent")
    print(f"{'='*60}")
    print(f"  Video-level acc: {va.mean():.4f} ± {va.std():.4f}")
    print(f"  Video-level  F1: {vf.mean():.4f} ± {vf.std():.4f}")
    print(f"  Video-level AUC: {np.nanmean(au):.4f}")
    print(f"\n  CNN MobileNetV2:        0.6156 acc")
    print(f"  Blink stride-12:        0.5933 acc")
    print(f"  Paper baseline (3-cls): ~0.65 acc")


if __name__ == "__main__":
    main()
