"""
Modelo TEMPORAL de parpadeo — GRU bidireccional + attention sobre la secuencia de
EAR (15fps), al estilo del HM-LSTM del paper (Ghoddoosian et al., 2019), pero más
liviano. Mismo protocolo: 5-fold CV subject-independent + agregación por video.

A diferencia de blink_features_dense_kfold.py (features agregadas por ventana),
acá el modelo ve la SECUENCIA cruda de parpadeos y aprende su dinámica temporal.

Uso:
    python temporal_blink_kfold.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

NPZ       = "data/processed/ear_dense.npz"
KEEP      = [0, 2]
LABEL_MAP = {0: 0, 2: 1}
K         = 5
FPS       = 15
WIN       = 450      # 30 s
STRIDE    = 150      # 10 s (ventanas solapadas → más muestras)
THR       = 0.21
SEED      = 42
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)


def frame_feats(ear: np.ndarray) -> np.ndarray:
    """Por frame: [EAR, |ΔEAR|, ojo_cerrado]."""
    vel = np.abs(np.diff(ear, prepend=ear[:1]))
    closed = (ear < THR).astype(np.float32)
    return np.stack([ear, vel, closed], axis=1)   # (T, 3)


def build():
    d = np.load(NPZ, allow_pickle=True)
    ear, subj, lbl, vid = d["ear"], d["subject"], d["label"], d["video_id"]
    X, y, S, V = [], [], [], []
    for v in np.unique(vid):
        m = (vid == v); cls = int(lbl[m][0])
        if cls not in KEEP:
            continue
        seq = frame_feats(ear[m].astype(np.float32))
        if len(seq) < WIN:
            continue
        s = subj[m][0]
        for st in range(0, len(seq) - WIN + 1, STRIDE):
            X.append(seq[st:st+WIN]); y.append(LABEL_MAP[cls]); S.append(s); V.append(v)
    X = np.asarray(X, np.float32)               # (N, WIN, 3)
    # normalizar EAR y velocidad (canales 0 y 1); el flag queda 0/1
    mu = X[:, :, :2].mean(axis=(0, 1)); sd = X[:, :, :2].std(axis=(0, 1)) + 1e-6
    X[:, :, :2] = (X[:, :, :2] - mu) / sd
    return X, np.array(y), np.array(S), np.array(V)


class BlinkGRU(nn.Module):
    def __init__(self, n_feat=3, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.gru  = nn.GRU(n_feat, hidden, layers, batch_first=True,
                           bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.attn = nn.Linear(hidden*2, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden*2, 2))

    def forward(self, x):
        out, _ = self.gru(x)                     # (B, T, 2H)
        w = torch.softmax(self.attn(out), dim=1) # (B, T, 1)
        ctx = (w * out).sum(dim=1)               # (B, 2H)
        return self.head(ctx)


def video_proba(proba, V, y):
    out = {}
    for v in np.unique(V):
        m = V == v; out[v] = (float(proba[m].mean()), int(y[m][0]))
    return out


def train_eval_fold(Xtr, ytr, Xva, yva, Vva, Xte, yte, Vte, epochs=40):
    model = BlinkGRU().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    best_va, best_state = -1, None

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), 64):
            idx = perm[i:i+64]
            xb = Xtr_t[idx].to(DEVICE); yb = ytr_t[idx].to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        # val a nivel de video para elegir el mejor epoch
        pv = predict(model, Xva)
        vp = video_proba(pv, Vva, yva)
        va = accuracy_score([t for _, t in vp.values()], [int(p >= .5) for p, _ in vp.values()])
        if va > best_va:
            best_va = va; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    pte = predict(model, Xte)
    return video_proba(pte, Vte, yte)


def predict(model, X):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.tensor(X[i:i+256]).to(DEVICE)
            out.append(torch.softmax(model(xb), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    X, y, S, V = build()
    print(f"Ventanas: {len(X):,}  forma: {X.shape[1:]}  "
          f"clases: {dict(zip(*np.unique(y, return_counts=True)))}")

    rng = np.random.default_rng(SEED)
    su = np.unique(S); rng.shuffle(su)
    folds = np.array_split(su, K)
    res = []

    for i in range(K):
        test_s = set(folds[i].tolist())
        rest = [s for s in su if s not in test_s]
        rng_f = np.random.default_rng(SEED+i); rng_f.shuffle(rest)
        val_s = set(rest[:3]); train_s = set(rest[3:])

        tr = np.array([s in train_s for s in S]); va = np.array([s in val_s for s in S]); te = np.array([s in test_s for s in S])
        vp = train_eval_fold(X[tr], y[tr], X[va], y[va], V[va], X[te], y[te], V[te])

        vt = np.array([t for _, t in vp.values()]); pr = np.array([p for p, _ in vp.values()])
        pred = (pr >= .5).astype(int)
        acc = accuracy_score(vt, pred); f1 = f1_score(vt, pred, average="macro", zero_division=0)
        auc = roc_auc_score(vt, pr) if len(set(vt)) > 1 else float("nan")
        print(f">> FOLD {i+1}: video_acc={acc:.4f}  video_F1={f1:.4f}  AUC={auc:.4f}  (n={len(vt)})")
        res.append({"fold": i, "video_accuracy": acc, "video_f1": f1, "video_auc": auc, "n_videos": len(vt)})

    va_ = np.array([r["video_accuracy"] for r in res]); vf = np.array([r["video_f1"] for r in res])
    au = np.array([r["video_auc"] for r in res])
    summary = {"model": "temporal_blink_gru", "k": K, "window_s": WIN/FPS, "folds": res,
               "video_acc_mean": float(va_.mean()), "video_acc_std": float(va_.std()),
               "video_f1_mean": float(vf.mean()), "video_auc_mean": float(np.nanmean(au))}
    Path("checkpoints/kfold").mkdir(parents=True, exist_ok=True)
    json.dump(summary, open("checkpoints/kfold/summary_temporal_blink.json", "w"), indent=2)

    print(f"\n{'='*60}")
    print(f"RESUMEN temporal-blink (GRU+attention) — {K}-fold CV")
    print(f"{'='*60}")
    print(f"  Video-level acc: {va_.mean():.4f} ± {va_.std():.4f}")
    print(f"  Video-level AUC: {np.nanmean(au):.4f}")
    print(f"\n  Parpadeo agregado (ref): 0.6917 ± 0.0425")
    print(f"  Paper HM-LSTM (3-cls):   0.6520 ± 0.0306")


if __name__ == "__main__":
    main()
