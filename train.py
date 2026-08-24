"""Train a small MLP on the recorded blendshape features.

On the train/val split
----------------------
Frames from one recording burst are almost identical to each other. If you
split randomly by frame, nearly every validation frame has a near-twin in the
training set and you will see 99% accuracy that completely falls apart on the
webcam. So the split here is by *session*: whole bursts go to train or to val,
never both. The number this prints is pessimistic-but-honest.
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

import config
from common import feature_dim


def load_data(path):
    labels, sessions, feats = [], [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0] == "label":
                continue
            labels.append(row[0])
            sessions.append(int(row[1]))
            feats.append([float(v) for v in row[2:]])
    if not feats:
        raise SystemExit(f"No data in {path}. Run collect.py first.")
    return np.asarray(feats, np.float32), np.asarray(labels), np.asarray(sessions)


def session_split(y, groups, val_frac, seed):
    """Hold out whole sessions, roughly val_frac of the sessions per class."""
    rng = np.random.default_rng(seed)
    val_sessions = set()
    by_label = defaultdict(set)
    for label, g in zip(y, groups):
        by_label[label].add(g)
    for label, sess in by_label.items():
        sess = np.array(sorted(sess))
        if len(sess) < 2:
            raise SystemExit(
                f"Class '{label}' only has {len(sess)} recording session(s). "
                "Record at least 2 (ideally 4+) separate bursts per expression "
                "so the split can hold one out.")
        k = max(1, min(len(sess) - 1, round(val_frac * len(sess))))
        val_sessions.update(rng.permutation(sess)[:k].tolist())
    is_val = np.array([g in val_sessions for g in groups])
    return ~is_val, is_val


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=(128, 64), p_drop=0.3):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.01,
                    help="gaussian jitter added to features during training")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    X, y_str, groups = load_data(config.DATA_CSV)

    # The feature width tells us whether hands were recorded; the checkpoint
    # carries that forward so run.py can never load a mismatched extractor.
    use_hands = X.shape[1] == feature_dim(True)
    if not use_hands and X.shape[1] != feature_dim(False):
        raise SystemExit(f"Unexpected feature width {X.shape[1]} in {config.DATA_CSV}")
    if use_hands != config.USE_HANDS:
        print(f"note: data was recorded with USE_HANDS={use_hands}, "
              f"config says {config.USE_HANDS}. Following the data.")

    classes = [c for c in config.EXPRESSIONS if c in set(y_str)]
    missing = set(y_str) - set(classes)
    if missing:
        print(f"warning: dropping samples with unknown labels {missing}")
        keep = np.isin(y_str, classes)
        X, y_str, groups = X[keep], y_str[keep], groups[keep]
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[c] for c in y_str])

    tr, va = session_split(y_str, groups, args.val_frac, args.seed)
    print(f"{len(X)} samples, {len(set(groups))} sessions, {len(classes)} classes, "
          f"{X.shape[1]} features (hands: {use_hands})")
    print(f"train {tr.sum()} / val {va.sum()}")

    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xtr = torch.from_numpy((X[tr] - mean) / std)
    Xva = torch.from_numpy((X[va] - mean) / std)
    ytr = torch.from_numpy(y[tr]).long()
    yva = torch.from_numpy(y[va]).long()

    counts = np.bincount(y[tr], minlength=len(classes)).astype(np.float32)
    weights = torch.from_numpy(counts.sum() / (len(classes) * np.maximum(counts, 1)))

    model = MLP(X.shape[1], len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    best_acc, best_state, patience = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            if len(b) < 2:
                continue
            xb = Xtr[b] + args.noise * torch.randn_like(Xtr[b])
            opt.zero_grad()
            loss = loss_fn(model(xb), ytr[b])
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xva).argmax(1)
            # macro accuracy: does not let a big class hide a broken small one
            accs = [(pred[yva == c] == c).float().mean().item()
                    for c in range(len(classes)) if (yva == c).any()]
            acc = float(np.mean(accs))
        if acc > best_acc:
            best_acc, patience = acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  loss {loss.item():.3f}  val macro-acc {acc:.3f}")
        if patience > 80:
            print("  early stop")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xva).argmax(1).numpy()
    yva_np = yva.numpy()

    print(f"\nbest val macro-accuracy: {best_acc:.3f}")
    print("\nper-class recall")
    for c, name in enumerate(classes):
        m = yva_np == c
        if m.any():
            print(f"  {name:<12} {(pred[m] == c).mean():.3f}  (n={m.sum()})")

    print("\nconfusion (rows = true, cols = predicted)")
    print(" " * 14 + "".join(f"{n[:8]:>10}" for n in classes))
    for c, name in enumerate(classes):
        row = [int(((yva_np == c) & (pred == p)).sum()) for p in range(len(classes))]
        print(f"  {name:<12}" + "".join(f"{v:>10}" for v in row))

    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes,
        "mean": mean,
        "std": std,
        "n_in": X.shape[1],
        "use_hands": use_hands,
    }, config.MODEL_PATH)
    print(f"\nsaved -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()