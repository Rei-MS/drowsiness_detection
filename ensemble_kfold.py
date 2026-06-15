"""
Ensemble CNN (apariencia) + blink-dense (dinámica de parpadeo), 5-fold CV
subject-independent a nivel de video. Misma partición de sujetos que entrenó
los checkpoints (make_kfold_splits, seed 42).

Por fold:
  - CNN: carga checkpoint del fold, inferencia → proba por video (mean softmax)
  - Blink: HistGBM entrenado en sujetos de train → proba por video
  - Ensemble: promedio ponderado → predicción por video

Uso:
    python ensemble_kfold.py --arch mobilenetv2
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import h5py
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data.dataset import make_kfold_splits, make_loaders
from src.models.backbone import DrowsinessModel
from blink_features_dense_kfold import build as build_blink

H5        = "/home/lilidl/drowsiness_crops.h5"
KEEP      = [0, 2]
LABEL_MAP = {0: 0, 2: 1}
K         = 5
SEED      = 42
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cnn_video_proba(ckpt_path, test_idx):
    """proba(clase1) por video = mean softmax sobre los frames del video."""
    _, _, test_loader = make_loaders(
        H5, test_idx[:1], test_idx[:1], test_idx,
        batch_size=256, num_workers=8, label_map=LABEL_MAP)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = DrowsinessModel(arch=ckpt["arch"], num_classes=ckpt["num_classes"], freeze=False)
    model.load_state_dict(ckpt["model_state"]); model.to(DEVICE).eval()

    probs = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            p = torch.softmax(model(imgs), dim=1)[:, 1].cpu().numpy()
            probs.append(p)
    probs = np.concatenate(probs)

    with h5py.File(H5, "r") as f:
        subj = f["subjects"][test_idx].astype(str)
        y    = f["y"][test_idx].astype(int)
    vids = np.array([f"{s}_{lbl}" for s, lbl in zip(subj, y)])

    out = {}
    for v in np.unique(vids):
        m = vids == v
        out[v] = (float(probs[m].mean()), LABEL_MAP[int(y[m][0])])
    return out


def main(arch, w_blink):
    # features de parpadeo (todas las ventanas, indexadas por sujeto/video)
    Xb, yb, Sb, Vb = build_blink()

    rows = {"cnn": [], "blink": [], "ens": []}
    for fold, tr_idx, va_idx, te_idx in make_kfold_splits(H5, k=K, val_subjects=3, keep_classes=KEEP):
        # sujetos de test de este fold
        with h5py.File(H5, "r") as f:
            test_subj = set(f["subjects"][te_idx].astype(str))

        # CNN
        cnn = cnn_video_proba(f"checkpoints/kfold/{arch}_fold{fold}.pt", te_idx)

        # Blink: train en sujetos NO-test, predecir test
        tr_m = np.array([s not in test_subj for s in Sb]); te_m = ~tr_m
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                             max_depth=4, l2_regularization=1.0, random_state=SEED)
        clf.fit(Xb[tr_m], yb[tr_m])
        pb = clf.predict_proba(Xb[te_m])[:, 1]
        Vte = Vb[te_m]; yte = yb[te_m]
        blink = {}
        for v in np.unique(Vte):
            m = Vte == v
            blink[v] = (float(pb[m].mean()), int(yte[m][0]))

        # videos comunes
        vids = sorted(set(cnn) & set(blink))
        yt   = np.array([cnn[v][1] for v in vids])
        pc   = np.array([cnn[v][0] for v in vids])
        pbk  = np.array([blink[v][0] for v in vids])
        pe   = (1 - w_blink) * pc + w_blink * pbk

        def metrics(p):
            pred = (p >= 0.5).astype(int)
            return (accuracy_score(yt, pred),
                    f1_score(yt, pred, average="macro", zero_division=0),
                    roc_auc_score(yt, p) if len(set(yt)) > 1 else float("nan"))

        for key, p in [("cnn", pc), ("blink", pbk), ("ens", pe)]:
            acc, f1, auc = metrics(p)
            rows[key].append((acc, f1, auc))
        a_c, _, _ = metrics(pc); a_b, _, _ = metrics(pbk); a_e, _, _ = metrics(pe)
        print(f"fold {fold+1}: CNN={a_c:.3f}  blink={a_b:.3f}  ENSEMBLE={a_e:.3f}  (n={len(vids)})")

    print(f"\n{'='*60}")
    print(f"ENSEMBLE {arch} + blink (w_blink={w_blink}) — 5-fold video-level")
    print(f"{'='*60}")
    summary = {"arch": arch, "w_blink": w_blink, "k": K}
    for key, name in [("cnn", f"CNN {arch}"), ("blink", "Blink denso"), ("ens", "ENSEMBLE")]:
        arr = np.array(rows[key])
        print(f"  {name:18s}  acc={arr[:,0].mean():.4f}±{arr[:,0].std():.4f}  "
              f"F1={arr[:,1].mean():.4f}  AUC={np.nanmean(arr[:,2]):.4f}")
        summary[key] = {"acc_mean": float(arr[:,0].mean()), "acc_std": float(arr[:,0].std()),
                        "f1_mean": float(arr[:,1].mean()), "auc_mean": float(np.nanmean(arr[:,2])),
                        "per_fold_acc": [float(x) for x in arr[:,0]]}
    Path("checkpoints/kfold").mkdir(parents=True, exist_ok=True)
    with open("checkpoints/kfold/summary_ensemble.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nGuardado: checkpoints/kfold/summary_ensemble.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="mobilenetv2")
    p.add_argument("--w-blink", type=float, default=0.5)
    a = p.parse_args()
    main(a.arch, a.w_blink)
