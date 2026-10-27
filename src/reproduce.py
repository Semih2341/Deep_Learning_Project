"""Milestone 1 - Reproduce the braindecode cropped-decoding tutorial on BCIC IV 2a.

Pipeline follows the braindecode example
`examples/model_building/plot_bcic_iv_2a_moabb_cropped.py`
(ShallowFBCSPNet, cropped training), with n_epochs raised from 4 to 100.

Two evaluation modes are supported:

* ``tutorial`` - exactly as in the example: the held-out session ("1test") is
  passed as the validation split and EarlyStopping(patience=10, load_best=True)
  restores the epoch with the lowest loss on it. This reproduces the reported
  number but selects the checkpoint on test data.
* ``clean``    - no validation split, no early stopping. The model is trained
  for a fixed number of epochs and the final weights are evaluated once on the
  held-out session. No decision is made using test data.

Usage:
    python src/reproduce.py --subjects 3 --mode tutorial
    python src/reproduce.py --subjects all --mode clean --seed 20200220
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from braindecode import EEGClassifier
from braindecode.datasets import MOABBDataset
from braindecode.models import ShallowFBCSPNet
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from braindecode.training import CroppedLoss, trial_preds_from_window_preds
from braindecode.util import set_random_seeds
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.helper import predefined_split

DATASET_NAME = "BNCI2014_001"
ALL_SUBJECTS = list(range(1, 10))

# Hyperparameters from the tutorial
LOW_CUT_HZ = 4.0  # tutorial; the paper's 73.7% uses 0-38 Hz (no high-pass)
HIGH_CUT_HZ = 38.0
FACTOR_NEW = 1e-3
INIT_BLOCK_SIZE = 1000
INPUT_WINDOW_SAMPLES = 1000
TRIAL_START_OFFSET_S = -0.5
LR = 0.0625 * 0.01
WEIGHT_DECAY = 0
BATCH_SIZE = 64
FINAL_CONV_LENGTH = 30
EARLY_STOPPING_PATIENCE = 10


def set_data_dir(data_dir):
    """Point MNE/MOABB downloads to a persistent folder (e.g. Google Drive)."""
    if data_dir is None:
        return
    import mne

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    for key in ("MNE_DATA", "MNE_DATASETS_BNCI_PATH"):
        mne.set_config(key, str(data_dir), set_env=True)


def load_dataset(subject_id, max_retries=5, wait_s=60):
    """Download/load one subject; the BNCI server is sometimes down, so retry."""
    for attempt in range(1, max_retries + 1):
        try:
            return MOABBDataset(dataset_name=DATASET_NAME, subject_ids=[subject_id])
        except Exception as err:  # network errors come in many types
            if attempt == max_retries:
                raise
            print(f"Download failed ({err!r}), retry {attempt}/{max_retries} in {wait_s}s")
            time.sleep(wait_s)


def preprocess_dataset(dataset, low_cut_hz=LOW_CUT_HZ):
    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: np.multiply(data, 1e6)),  # V -> uV
        Preprocessor("filter", l_freq=low_cut_hz or None, h_freq=HIGH_CUT_HZ),
        Preprocessor(
            exponential_moving_standardize,
            factor_new=FACTOR_NEW,
            init_block_size=INIT_BLOCK_SIZE,
        ),
    ]
    preprocess(dataset, preprocessors, n_jobs=-1)
    return dataset


def build_model(n_chans, n_classes, cuda):
    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        n_times=INPUT_WINDOW_SAMPLES,
        final_conv_length=FINAL_CONV_LENGTH,
    )
    if cuda:
        model.cuda()
    model.to_dense_prediction_model()
    return model


def make_windows(dataset, n_preds_per_input):
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    assert all(ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets)
    return create_windows_from_events(
        dataset,
        trial_start_offset_samples=int(TRIAL_START_OFFSET_S * sfreq),
        trial_stop_offset_samples=0,
        window_size_samples=INPUT_WINDOW_SAMPLES,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        preload=True,
    )


def trialwise_accuracy(clf, windows_set):
    """Average crop predictions within each trial, then compare to the label."""
    window_preds = clf.forward(windows_set, training=False).cpu().numpy()
    md = windows_set.get_metadata()
    trial_preds = trial_preds_from_window_preds(
        window_preds,
        md["i_window_in_trial"].to_numpy(),
        md["i_stop_in_trial"].to_numpy(),
    )
    y_pred = np.array([p.mean(axis=1).argmax() for p in trial_preds])
    y_true = md.loc[md["i_window_in_trial"] == 0, "target"].to_numpy()
    assert len(y_pred) == len(y_true)
    return float((y_pred == y_true).mean()), y_true.tolist(), y_pred.tolist()


def run_subject(subject_id, mode, n_epochs, seed, data_dir=None, low_cut_hz=LOW_CUT_HZ):
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    if cuda:
        torch.backends.cudnn.benchmark = True
    set_random_seeds(seed=seed, cuda=cuda)

    set_data_dir(data_dir)
    dataset = preprocess_dataset(load_dataset(subject_id), low_cut_hz)

    n_classes = 4
    n_chans = dataset[0][0].shape[0]
    model = build_model(n_chans, n_classes, cuda)
    n_preds_per_input = model.get_output_shape()[2]

    windows = make_windows(dataset, n_preds_per_input)
    splits = windows.split("session")
    train_set, test_set = splits["0train"], splits["1test"]

    callbacks = [
        "accuracy",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=max(1, n_epochs - 1))),
    ]
    if mode == "tutorial":
        train_split = predefined_split(test_set)
        callbacks.append(
            ("early_stopping", EarlyStopping(patience=EARLY_STOPPING_PATIENCE, load_best=True))
        )
    elif mode == "clean":
        train_split = None
    else:
        raise ValueError(f"Unknown mode: {mode}")

    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=F.cross_entropy,
        optimizer=torch.optim.AdamW,
        train_split=train_split,
        optimizer__lr=LR,
        optimizer__weight_decay=WEIGHT_DECAY,
        iterator_train__shuffle=True,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        device=device,
        classes=list(range(n_classes)),
        max_epochs=n_epochs,
    )

    t0 = time.time()
    clf.fit(train_set, y=None, epochs=n_epochs)
    train_time = time.time() - t0

    test_acc, y_true, y_pred = trialwise_accuracy(clf, test_set)
    train_acc, _, _ = trialwise_accuracy(clf, train_set)

    history_keys = [k for k in clf.history[0].keys() if k != "batches"]
    history = [{k: row.get(k) for k in history_keys} for row in clf.history]
    if mode == "tutorial":
        best_epoch = int(np.argmin([h["valid_loss"] for h in history])) + 1
    else:
        best_epoch = len(history)

    return {
        "subject": subject_id,
        "mode": mode,
        "seed": seed,
        "low_cut_hz": low_cut_hz,
        "n_epochs_max": n_epochs,
        "n_epochs_run": len(history),
        "selected_epoch": best_epoch,
        "test_acc": test_acc,
        "train_acc": train_acc,
        "train_time_s": train_time,
        "y_true": y_true,
        "y_pred": y_pred,
        "history": history,
        "env": environment_info(device),
    }


def environment_info(device):
    import braindecode
    import mne
    import moabb
    import skorch

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "braindecode": braindecode.__version__,
        "moabb": moabb.__version__,
        "mne": mne.__version__,
        "skorch": skorch.__version__,
        "device": device,
    }
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def _to_json(obj):
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    return str(obj)


def parse_subjects(arg):
    if arg == "all":
        return ALL_SUBJECTS
    return [int(s) for s in arg.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subjects", default="3", help='e.g. "3", "1,2,3" or "all"')
    parser.add_argument("--mode", choices=["tutorial", "clean"], default="tutorial")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20200220)
    parser.add_argument("--low-cut", type=float, default=LOW_CUT_HZ, help="high-pass in Hz, 0 = none")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--data-dir", default=None, help="persistent MNE/MOABB data folder")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_name = f"{args.mode}_seed{args.seed}"
    if args.low_cut != LOW_CUT_HZ:
        run_name += f"_{args.low_cut:g}-{HIGH_CUT_HZ:g}Hz"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for subject_id in parse_subjects(args.subjects):
        out_file = out_dir / f"subject_{subject_id:02d}.json"
        if out_file.exists() and not args.overwrite:
            # lets a disconnected Colab session resume where it stopped
            print(f"Skipping subject {subject_id}: {out_file} exists")
            continue
        print(f"\n=== Subject {subject_id} | {run_name} ===")
        result = run_subject(
            subject_id, args.mode, args.epochs, args.seed, args.data_dir, args.low_cut
        )
        out_file.write_text(json.dumps(result, indent=1, default=_to_json))
        print(
            f"Subject {subject_id}: test acc = {100 * result['test_acc']:.2f}% "
            f"(epoch {result['selected_epoch']}/{result['n_epochs_run']})"
        )


if __name__ == "__main__":
    main()
