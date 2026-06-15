"""
Evaluación multi-semilla del modelo temporal de parpadeo (GRU+attention).
Promediar sobre semillas evita reportar una semilla afortunada (el 0.717 inicial
fue suerte). Guarda checkpoints/kfold/summary_temporal_blink.json.

Uso: python run_temporal_multiseed.py
"""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import temporal_blink_kfold as T

SEEDS = [0, 1, 2, 3, 4]
K = 5


def main():
    X, y, S, V = T.build()
    print(f"Ventanas: {len(X):,}  forma: {X.shape[1:]}")

    per_seed_mean, per_seed_folds, all_fold_acc, all_fold_auc = [], [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        rng = np.random.default_rng(seed); su = np.unique(S); rng.shuffle(su)
        folds = np.array_split(su, K); accs = []
        for i in range(K):
            ts = set(folds[i].tolist()); rest = [s for s in su if s not in ts]
            rf = np.random.default_rng(seed*10+i); rf.shuffle(rest)
            vs = set(rest[:3]); trs = set(rest[3:])
            tr = np.array([s in trs for s in S]); va = np.array([s in vs for s in S]); te = np.array([s in ts for s in S])
            vp = T.train_eval_fold(X[tr], y[tr], X[va], y[va], V[va], X[te], y[te], V[te], epochs=40)
            vt = np.array([t for _, t in vp.values()]); pr = np.array([p for p, _ in vp.values()])
            acc = accuracy_score(vt, (pr >= .5).astype(int))
            auc = roc_auc_score(vt, pr) if len(set(vt)) > 1 else float("nan")
            accs.append(acc); all_fold_acc.append(acc); all_fold_auc.append(auc)
        m = float(np.mean(accs)); per_seed_mean.append(m); per_seed_folds.append([round(a, 4) for a in accs])
        print(f"seed {seed}: {m:.4f}  folds={[round(a,2) for a in accs]}")

    psm = np.array(per_seed_mean)
    summary = {
        "model": "temporal_blink_gru", "k": K, "seeds": SEEDS,
        "per_seed_mean": [round(x, 4) for x in per_seed_mean],
        "per_seed_folds": per_seed_folds,
        "acc_mean": float(psm.mean()), "acc_std": float(psm.std()),
        "best_seed_mean": float(psm.max()),
        "all_fold_acc": [round(x, 4) for x in all_fold_acc],
        "auc_mean": float(np.nanmean(all_fold_auc)),
    }
    Path("checkpoints/kfold").mkdir(parents=True, exist_ok=True)
    json.dump(summary, open("checkpoints/kfold/summary_temporal_blink.json", "w"), indent=2)

    print(f"\n{'='*60}")
    print(f"TEMPORAL multi-semilla ({len(SEEDS)} seeds × {K} folds)")
    print(f"{'='*60}")
    print(f"  acc (media de semillas): {psm.mean():.4f} ± {psm.std():.4f}")
    print(f"  mejor semilla:           {psm.max():.4f}  (¡no reportar esto solo!)")
    print(f"  Parpadeo agregado (ref): 0.6917 ± 0.0425")
    print(f"  Paper HM-LSTM (3-cls):   0.6520 ± 0.0306")


if __name__ == "__main__":
    main()
