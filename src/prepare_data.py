"""Milestone 1 - Data preparation: build the training, validation and test sets.

Uses exactly the same preprocessing and windowing as `reproduce.py`, and splits each
subject's data into three parts without ever using the test session for a decision:

* train - session `0train`, runs 0-4 (240 trials)
* valid - session `0train`, run 5   (48 trials; held out from training, for model selection)
* test  - session `1test`, all runs (288 trials; used only for the final evaluation)

The validation set is a whole run, so no crop of a validation trial overlaps a training trial.
For each subject, the script reports the shapes and class counts of every split. It saves them
to `results/data_splits.json` and can optionally save the arrays as `.npz` files.

Usage:
    python src/prepare_data.py --subjects all
    python src/prepare_data.py --subjects 3 --save-npz data/
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from reproduce import (
    HIGH_CUT_HZ,
    INPUT_WINDOW_SAMPLES,
    LOW_CUT_HZ,
    build_model,
    load_dataset,
    make_windows,
    parse_subjects,
    preprocess_dataset,
    set_data_dir,
)

CLASS_NAMES = ["feet", "left_hand", "right_hand", "tongue"]  # braindecode label order 0..3
VALID_RUN = "5"


def split_windows(windows):
    """Split a subject's windows into train / valid / test (see module docstring)."""
    by_session = windows.split("session")
    train_session = by_session["0train"]
    runs = train_session.split("run")
    train = [ds for run, ds in runs.items() if run != VALID_RUN]
    from braindecode.datasets import BaseConcatDataset

    return {
        "train": BaseConcatDataset(train),
        "valid": runs[VALID_RUN],
        "test": by_session["1test"],
    }


def describe(split):
    """Shapes and class counts of one split (inputs X, outputs y)."""
    md = split.get_metadata()
    trials = md[md["i_window_in_trial"] == 0]
    x0, _, _ = split[0]
    return {
        "n_trials": int(len(trials)),
        "n_windows": int(len(split)),
        "X_shape": [int(len(split))] + list(x0.shape),  # (windows, channels, samples)
        "y_shape": [int(len(split))],
        "trials_per_class": {
            CLASS_NAMES[k]: int(v) for k, v in sorted(Counter(trials["target"]).items())
        },
    }


def to_arrays(split):
    X = np.stack([split[i][0] for i in range(len(split))]).astype(np.float32)
    y = np.array([split[i][1] for i in range(len(split))], dtype=np.int64)
    return X, y


def prepare_subject(subject_id, low_cut_hz=LOW_CUT_HZ):
    dataset = preprocess_dataset(load_dataset(subject_id), low_cut_hz)
    n_chans = dataset[0][0].shape[0]
    # the model fixes how many crop predictions one window gives (window stride)
    model = build_model(n_chans, len(CLASS_NAMES), cuda=False)
    n_preds_per_input = model.get_output_shape()[2]
    windows = make_windows(dataset, n_preds_per_input)
    return split_windows(windows), n_preds_per_input


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subjects", default="all", help='e.g. "3", "1,2,3" or "all"')
    parser.add_argument("--low-cut", type=float, default=LOW_CUT_HZ, help="high-pass in Hz, 0 = none")
    parser.add_argument("--data-dir", default=None, help="persistent MNE/MOABB data folder")
    parser.add_argument("--out-file", default="results/data_splits.json")
    parser.add_argument("--save-npz", default=None, help="folder for X/y arrays (not committed)")
    args = parser.parse_args()

    set_data_dir(args.data_dir)
    summary = {
        "band_hz": [args.low_cut, HIGH_CUT_HZ],
        "window_samples": INPUT_WINDOW_SAMPLES,
        "split": {
            "train": "session 0train, runs 0-4",
            "valid": f"session 0train, run {VALID_RUN}",
            "test": "session 1test, runs 0-5",
        },
        "subjects": {},
    }
    for subject_id in parse_subjects(args.subjects):
        splits, n_preds = prepare_subject(subject_id, args.low_cut)
        info = {name: describe(split) for name, split in splits.items()}
        info["crops_per_window"] = int(n_preds)
        summary["subjects"][subject_id] = info
        print(f"Subject {subject_id}: " + ", ".join(
            f"{name} X{tuple(d['X_shape'])} ({d['n_trials']} trials)"
            for name, d in info.items() if isinstance(d, dict)))
        if args.save_npz:
            out = Path(args.save_npz)
            out.mkdir(parents=True, exist_ok=True)
            for name, split in splits.items():
                X, y = to_arrays(split)
                np.savez_compressed(out / f"subject_{subject_id:02d}_{name}.npz", X=X, y=y)

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_file).write_text(json.dumps(summary, indent=1))
    print(f"Wrote {args.out_file}")


if __name__ == "__main__":
    main()
