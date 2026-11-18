"""Milestone 2 - Improve: controlled experiments on top of the M1 baseline.

Every run uses the same data split as `prepare_data.py`:

* within-subject: train = session 0train runs 0-4, valid = run 5, test = session 1test
* LOSO (leave one subject out), fold k: train = runs 0-4 of the other 8 subjects,
  valid = run 5 of the other 8 subjects, test = session 1test of subject k

Every model trains for the same number of epochs (default 100). The weights from the epoch with
the best **validation** accuracy are restored, and then the test set is evaluated once (primary
result). As a secondary result, the test accuracy of the weights after the full budget is also
stored (the M1 "clean" rule, fixed in advance). Test data is never used for any decision.

Options (one factor at a time):
    --model shallow|eegnet|conformer   architecture
    --ea                               Euclidean alignment per subject and session
    --augment                          random augmentation of training batches

Inputs are the cached arrays written by
`python src/prepare_data.py --subjects all --save-npz <data-npz>`.

Usage:
    python src/improve.py --protocol within --model shallow --seed 1 --data-npz ../data_npz
    python src/improve.py --protocol loso --model eegnet --ea --seed 1 --data-npz ../data_npz
"""

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ALL_SUBJECTS = list(range(1, 10))
N_CLASSES = 4
SFREQ = 250.0
WINDOWS_PER_TRIAL = 2  # prepare_data.py: each 1125-sample trial gives 2 windows of 1000 samples

# Learning rates follow each model's original paper / braindecode example
MODEL_CONFIG = {
    "shallow": {"lr": 0.0625 * 0.01, "weight_decay": 0.0, "betas": (0.9, 0.999)},
    "eegnet": {"lr": 1e-3, "weight_decay": 0.0, "betas": (0.9, 0.999)},
    "conformer": {"lr": 2e-4, "weight_decay": 0.0, "betas": (0.5, 0.999)},
}


# ---------------------------------------------------------------- data
def load_subject(data_npz, subject):
    out = {}
    for split in ("train", "valid", "test"):
        d = np.load(Path(data_npz) / f"subject_{subject:02d}_{split}.npz")
        out[split] = (d["X"].astype(np.float32), d["y"].astype(np.int64))
    return out


def ea_matrix(X):
    """R^-1/2, where R is the mean spatial covariance of the windows (no labels used)."""
    covs = np.einsum("nct,ndt->ncd", X, X) / X.shape[2]
    R = covs.mean(axis=0)
    w, V = np.linalg.eigh(R)
    return (V / np.sqrt(w)) @ V.T


def euclidean_align(subject_data):
    """Euclidean alignment (He & Wu, 2019), computed separately per recording session.

    Session 0train = train + valid runs; session 1test = test. Only the unlabeled EEG of
    each session is used to compute the reference matrix.
    """
    Xtr, ytr = subject_data["train"]
    Xva, yva = subject_data["valid"]
    Xte, yte = subject_data["test"]
    W_train_session = ea_matrix(np.concatenate([Xtr, Xva]))
    W_test_session = ea_matrix(Xte)
    align = lambda W, X: np.einsum("cd,ndt->nct", W, X).astype(np.float32)
    return {
        "train": (align(W_train_session, Xtr), ytr),
        "valid": (align(W_train_session, Xva), yva),
        "test": (align(W_test_session, Xte), yte),
    }


def build_splits(args, test_subject):
    """Return train / valid / test arrays for one subject (within) or one fold (loso)."""
    subjects = {s: load_subject(args.data_npz, s) for s in
                (ALL_SUBJECTS if args.protocol == "loso" else [test_subject])}
    if args.ea:
        subjects = {s: euclidean_align(d) for s, d in subjects.items()}
    if args.protocol == "within":
        return subjects[test_subject]
    others = [s for s in ALL_SUBJECTS if s != test_subject]
    cat = lambda split: tuple(np.concatenate([subjects[s][split][i] for s in others]) for i in (0, 1))
    return {"train": cat("train"), "valid": cat("valid"), "test": subjects[test_subject]["test"]}


# ---------------------------------------------------------------- model / augmentation
def build_model(name, n_chans, n_times):
    from braindecode.models import EEGConformer, EEGNet, ShallowFBCSPNet

    if name == "shallow":
        return ShallowFBCSPNet(n_chans=n_chans, n_outputs=N_CLASSES, n_times=n_times,
                               final_conv_length="auto")
    if name == "eegnet":
        return EEGNet(n_chans=n_chans, n_outputs=N_CLASSES, n_times=n_times)
    if name == "conformer":
        return EEGConformer(n_chans=n_chans, n_outputs=N_CLASSES, n_times=n_times)
    raise ValueError(name)


def build_augmentation(seed):
    """Label-preserving augmentations, each applied to a training sample with p = 0.5.

    ChannelsSymmetry is deliberately not used: mirroring the hemispheres would turn a
    left-hand trial into a right-hand one without changing its label.
    """
    from braindecode.augmentation import FrequencyShift, GaussianNoise, SmoothTimeMask

    rng = np.random.RandomState(seed)
    transforms = [
        FrequencyShift(probability=0.5, sfreq=SFREQ, max_delta_freq=2, random_state=rng),
        SmoothTimeMask(probability=0.5, mask_len_samples=100, random_state=rng),
        GaussianNoise(probability=0.5, std=0.1, random_state=rng),
    ]

    def apply(X, y):
        for t in transforms:
            X, y = t(X, y)
        return X, y

    return apply


# ---------------------------------------------------------------- training
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, X, y, batch_size=256):
    """Loss and trial-wise accuracy (window probabilities averaged within each trial)."""
    model.eval()
    logits = torch.cat([model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])
    loss = nn.functional.cross_entropy(logits, y).item()
    probs = logits.softmax(dim=1).reshape(-1, WINDOWS_PER_TRIAL, N_CLASSES).mean(dim=1)
    y_trial = y.reshape(-1, WINDOWS_PER_TRIAL)[:, 0]
    y_pred = probs.argmax(dim=1)
    acc = (y_pred == y_trial).float().mean().item()
    return loss, acc, y_trial.cpu().tolist(), y_pred.cpu().tolist()


def run_unit(args, test_subject, device):
    set_seed(args.seed)
    data = build_splits(args, test_subject)
    to_t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (
        (to_t(data[s][0]), to_t(data[s][1], torch.long)) for s in ("train", "valid", "test"))

    model = build_model(args.model, Xtr.shape[1], Xtr.shape[2]).to(device)
    cfg = MODEL_CONFIG[args.model]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"], betas=cfg["betas"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    augment = build_augmentation(args.seed) if args.augment else None

    best = {"acc": -1.0, "loss": np.inf, "epoch": 0, "state": None}
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        losses = []
        for i in range(0, len(Xtr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            if augment is not None:
                xb, yb = augment(xb, yb)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        va_loss, va_acc, _, _ = evaluate(model, Xva, yva)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "valid_loss": va_loss, "valid_acc": va_acc})
        # model selection on validation only: best accuracy, ties broken by lower loss
        if va_acc > best["acc"] or (va_acc == best["acc"] and va_loss < best["loss"]):
            best = {"acc": va_acc, "loss": va_loss, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    train_time = time.time() - t0

    # Secondary, also test-free rule fixed in advance: the weights after the full budget
    # (the M1 "clean" protocol). Reported next to the primary, validation-selected result.
    _, last_acc, _, last_pred = evaluate(model, Xte, yte)

    model.load_state_dict(best["state"])
    _, train_acc, _, _ = evaluate(model, Xtr, ytr)
    test_loss, test_acc, y_true, y_pred = evaluate(model, Xte, yte)  # single final evaluation
    return {
        "protocol": args.protocol, "model": args.model, "ea": args.ea, "augment": args.augment,
        "seed": args.seed, "test_subject": test_subject, "epochs": args.epochs,
        "batch_size": args.batch_size, "optimizer": {k: list(v) if isinstance(v, tuple) else v
                                                     for k, v in cfg.items()},
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_train_windows": len(Xtr), "n_valid_windows": len(Xva), "n_test_windows": len(Xte),
        "best_valid_epoch": best["epoch"], "valid_acc": best["acc"],
        "train_acc": train_acc, "test_acc": test_acc, "test_loss": test_loss,
        "test_acc_last_epoch": last_acc, "y_pred_last_epoch": last_pred,
        "train_time_s": train_time, "y_true": y_true, "y_pred": y_pred,
        "history": history, "env": environment_info(device),
    }


def environment_info(device):
    import braindecode
    import mne

    info = {"python": platform.python_version(), "torch": torch.__version__,
            "braindecode": braindecode.__version__, "mne": mne.__version__, "device": device}
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def run_name(args):
    return (f"{args.protocol}_{args.model}" + ("_ea" if args.ea else "")
            + ("_aug" if args.augment else "") + f"_seed{args.seed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--protocol", choices=["within", "loso"], default="within")
    parser.add_argument("--model", choices=list(MODEL_CONFIG), default="shallow")
    parser.add_argument("--ea", action="store_true", help="Euclidean alignment")
    parser.add_argument("--augment", action="store_true", help="data augmentation")
    parser.add_argument("--subjects", default="all", help='test subjects, e.g. "3" or "all"')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20200220)
    parser.add_argument("--data-npz", default="../data_npz")
    parser.add_argument("--out-dir", default="results_m2")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    subjects = ALL_SUBJECTS if args.subjects == "all" else [int(s) for s in args.subjects.split(",")]
    out_dir = Path(args.out_dir) / run_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in subjects:
        out_file = out_dir / f"subject_{s:02d}.json"
        if out_file.exists() and not args.overwrite:
            print(f"skip {out_file}")
            continue
        r = run_unit(args, s, device)
        out_file.write_text(json.dumps(r, indent=1))
        print(f"{run_name(args)} subject {s}: test {100 * r['test_acc']:.2f}% "
              f"(last epoch {100 * r['test_acc_last_epoch']:.2f}%) "
              f"(valid {100 * r['valid_acc']:.2f}% @ epoch {r['best_valid_epoch']}, "
              f"{r['train_time_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
