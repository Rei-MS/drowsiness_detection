"""
Modelo sobre features de EAR/PERCLOS (libre de apariencia) — paso 1.

Ventanas deslizantes sobre la secuencia de EAR de cada video → features
estadísticas → HistGradientBoosting con 5-fold CV subject-independent +
agregación por video. Comparación honesta contra el CNN (0.62 video acc).

NOTA: EAR muestreado a stride=12 (2.5 fps) → PERCLOS y stats de EAR son
confiables; dinámica fina de parpadeo NO (submuestreada). Ver paso 2.

Uso:
    python blink_features_kfold.py
"""

import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

H5        = "/home/lilidl/drowsiness_crops.h5"
EAR_NPY   = "data/processed/ear_dlib.npy"
KEEP      = [0, 2]
LABEL_MAP = {0: 0, 2: 1}
K         = 5
WIN       = 64       # ~25 s a 2.5 fps
STRIDE    = 16
EAR_THR   = 0.21     # umbral ojo cerrado
SEED      = 42


def window_features(ear_seq: np.ndarray) -> np.ndarray:
    """Vector de features sobre una ventana de EAR (libre de apariencia)."""
    e = ear_seq
    d = np.abs(np.diff(e)) if len(e) > 1 else np.array([0.0])
    closed = e < EAR_THR
    # transiciones abierto→cerrado = proxy de parpadeo (limitado por el muestreo)
    blink_starts = int(np.sum((~closed[:-1]) & (closed[1:]))) if len(e) > 1 else 0
    return np.array([
        e.mean(), e.std(), e.min(), e.max(), np.median(e),
        np.percentile(e, 10), np.percentile(e, 25),
        closed.mean(),                 # PERCLOS
        (e < 0.25).mean(),             # PERCLOS umbral laxo
        d.mean(), d.max(), d.std(),    # velocidad de EAR
        blink_starts / len(e),         # tasa de cierres por frame
    ], dtype=np.float32)


def build_dataset():
    ear = np.load(EAR_NPY)
    with h5py.File(H5, "r") as f:
        y    = f["y"][:]
        subj = f["subjects"][:].astype(str)

    # videos = (sujeto, clase original) contiguos
    keep = np.isin(y, KEEP) & np.isfinite(ear)
    X_feat, y_w, subj_w, vid_w = [], [], [], []

    for s in np.unique(subj):
        for cls in KEEP:
            m = (subj == s) & (y == cls) & np.isfinite(ear)
            if m.sum() < WIN:
                continue
            seq = ear[m]
            vid = f"{s}_{cls}"
            for start in range(0, len(seq) - WIN + 1, STRIDE):
                X_feat.append(window_features(seq[start:start + WIN]))
                y_w.append(LABEL_MAP[cls])
                subj_w.append(s)
                vid_w.append(vid)

    return (np.array(X_feat), np.array(y_w),
            np.array(subj_w), np.array(vid_w))


def kfold_subjects(subjects_unique, k, seed):
    rng = np.random.default_rng(seed)
    su = subjects_unique.copy()
    rng.shuffle(su)
    return np.array_split(su, k)


def main():
    X, y, subj_w, vid_w = build_dataset()
    print(f"Ventanas: {len(X):,}  features: {X.shape[1]}  "
          f"clases: {dict(zip(*np.unique(y, return_counts=True)))}")

    unique_subj = np.unique(subj_w)
    folds = kfold_subjects(unique_subj, K, SEED)
    results = []

    for i in range(K):
        test_s  = set(folds[i].tolist())
        train_m = np.array([s not in test_s for s in subj_w])
        test_m  = ~train_m

        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, random_state=SEED,
        )
        clf.fit(X[train_m], y[train_m])

        proba = clf.predict_proba(X[test_m])[:, 1]
        pred  = (proba >= 0.5).astype(int)
        win_f1 = f1_score(y[test_m], pred, average="macro", zero_division=0)

        # agregación por video: promedio de probabilidad
        vids = vid_w[test_m]; yt = y[test_m]
        vt, vp = [], []
        for v in np.unique(vids):
            vm = vids == v
            vt.append(int(yt[vm][0]))
            vp.append(int(proba[vm].mean() >= 0.5))
        vt, vp = np.array(vt), np.array(vp)
        vid_acc = accuracy_score(vt, vp)
        vid_f1  = f1_score(vt, vp, average="macro", zero_division=0)

        print(f">> FOLD {i+1}: win_F1={win_f1:.4f}  "
              f"video_acc={vid_acc:.4f}  video_F1={vid_f1:.4f}  "
              f"(n_videos={len(vt)}, test_subj={len(test_s)})")
        results.append({"fold": i, "win_f1": float(win_f1),
                        "video_accuracy": float(vid_acc), "video_f1": float(vid_f1),
                        "n_videos": len(vt)})

    wf = np.array([r["win_f1"] for r in results])
    va = np.array([r["video_accuracy"] for r in results])
    vf = np.array([r["video_f1"] for r in results])

    summary = {
        "model": "blink_features_histgbm", "k": K, "window": WIN, "stride": STRIDE,
        "folds": results,
        "win_f1_mean": float(wf.mean()), "win_f1_std": float(wf.std()),
        "video_acc_mean": float(va.mean()), "video_acc_std": float(va.std()),
        "video_f1_mean": float(vf.mean()), "video_f1_std": float(vf.std()),
    }
    Path("checkpoints/kfold").mkdir(parents=True, exist_ok=True)
    with open("checkpoints/kfold/summary_blink_features.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESUMEN blink-features (PERCLOS+EAR) — {K}-fold CV subject-independent")
    print(f"{'='*60}")
    print(f"  Window-level F1: {wf.mean():.4f} ± {wf.std():.4f}")
    print(f"  Video-level acc: {va.mean():.4f} ± {va.std():.4f}")
    print(f"  Video-level  F1: {vf.mean():.4f} ± {vf.std():.4f}")
    print(f"\n  CNN MobileNetV2 (ref): 0.6156 video acc")
    print(f"  Paper baseline (3-cls): ~0.65 acc")


if __name__ == "__main__":
    main()
