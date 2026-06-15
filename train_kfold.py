"""
5-fold cross-validation subject-independent — el protocolo del paper UTA-RLDD.

Entrena un CNN fresco en cada fold, evalúa frame-level F1 y video-level accuracy,
y reporta media ± std sobre los 5 folds. Es la forma honesta de comparar con el paper.

Uso:
    source /home/lilidl/pnl/.venv/bin/activate
    cd /home/lilidl/tp_final_cv2
    python train_kfold.py --arch mobilenetv2
    python train_kfold.py --arch resnet50v2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.data.dataset import (make_kfold_splits, make_loaders, video_level_eval)
from src.models.backbone import DrowsinessModel
from src.training.train import train_model

H5           = "/home/lilidl/drowsiness_crops.h5"
KEEP_CLASSES = [0, 2]
LABEL_MAP    = {0: 0, 2: 1}
CLASS_NAMES  = ["alerta", "somnoliento"]
K            = 5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_fold(model, test_loader, test_idx):
    model.to(DEVICE).eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            preds.append(model(imgs).argmax(1).cpu().numpy())
            labels.append(lbls.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)

    frame_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    video    = video_level_eval(H5, test_idx, preds, label_map=LABEL_MAP)
    return frame_f1, video


def main(arch: str):
    Path("checkpoints/kfold").mkdir(parents=True, exist_ok=True)
    results = []

    for fold, train_idx, val_idx, test_idx in make_kfold_splits(
        H5, k=K, val_subjects=3, keep_classes=KEEP_CLASSES
    ):
        print(f"\n{'#'*60}")
        print(f"# FOLD {fold + 1}/{K} — {arch}")
        print(f"# train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")
        print(f"{'#'*60}")

        train_loader, val_loader, test_loader = make_loaders(
            H5, train_idx, val_idx, test_idx,
            batch_size=128, num_workers=8, label_map=LABEL_MAP,
        )

        model = DrowsinessModel(arch=arch, num_classes=2, freeze=True, dropout=0.2)
        train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            checkpoint_path=Path(f"checkpoints/kfold/{arch}_fold{fold}.pt"),
            epochs_frozen=10, epochs_unfrozen=20,
            lr_head=1e-3, lr_finetune=1e-4, patience=8, unfreeze_blocks=2,
        )

        # cargar mejor checkpoint del fold antes de evaluar test
        ckpt = torch.load(f"checkpoints/kfold/{arch}_fold{fold}.pt",
                          map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])

        frame_f1, video = eval_fold(model, test_loader, test_idx)
        print(f"\n>> FOLD {fold+1} — frame_F1={frame_f1:.4f}  "
              f"video_acc={video['video_accuracy']:.4f}  "
              f"video_F1={video['video_f1']:.4f}  (n_videos={video['n_videos']})")

        results.append({
            "fold": fold,
            "frame_f1": frame_f1,
            "video_accuracy": video["video_accuracy"],
            "video_f1": video["video_f1"],
            "n_videos": video["n_videos"],
            "val_f1_best": ckpt["val_f1"],
        })

    # ── Resumen ───────────────────────────────────────────────────────────────
    frame_f1s = np.array([r["frame_f1"] for r in results])
    vid_accs  = np.array([r["video_accuracy"] for r in results])
    vid_f1s   = np.array([r["video_f1"] for r in results])

    summary = {
        "arch": arch,
        "k": K,
        "folds": results,
        "frame_f1_mean": float(frame_f1s.mean()), "frame_f1_std": float(frame_f1s.std()),
        "video_acc_mean": float(vid_accs.mean()),  "video_acc_std": float(vid_accs.std()),
        "video_f1_mean": float(vid_f1s.mean()),     "video_f1_std": float(vid_f1s.std()),
    }

    with open(f"checkpoints/kfold/summary_{arch}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESUMEN {arch} — {K}-fold CV subject-independent")
    print(f"{'='*60}")
    print(f"  Frame-level  F1: {frame_f1s.mean():.4f} ± {frame_f1s.std():.4f}")
    print(f"  Video-level acc: {vid_accs.mean():.4f} ± {vid_accs.std():.4f}")
    print(f"  Video-level  F1: {vid_f1s.mean():.4f} ± {vid_f1s.std():.4f}")
    print(f"\n  Paper baseline (3-clases): ~0.65 acc")
    print(f"  Guardado: checkpoints/kfold/summary_{arch}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="mobilenetv2",
                   choices=["mobilenetv2", "resnet50v2"])
    main(p.parse_args().arch)
