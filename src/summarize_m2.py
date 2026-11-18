"""Collect Milestone 2 result files into the results table (results_m2/summary.{md,csv}).

For each experiment: the mean over the 9 subjects (or LOSO folds) is computed per seed, and
the table reports mean ± std of that value over seeds. Validation accuracy is shown next to the
test accuracy, because the model is chosen on validation.

Usage:
    python src/summarize_m2.py --results-dir results_m2
"""

import argparse
import sys
import json
from pathlib import Path

import pandas as pd

# id, protocol, model, ea, augment, what changed vs. the protocol's baseline
EXPERIMENTS = [
    ("B0", "within", "shallow", False, False, "baseline: ShallowFBCSPNet (M1 model, trial-wise, valid-selected)"),
    ("E1", "within", "eegnet", False, False, "model → EEGNet"),
    ("E2", "within", "conformer", False, False, "model → EEGConformer"),
    ("E3", "within", "shallow", False, True, "+ data augmentation"),
    ("E4", "within", "shallow", True, False, "+ Euclidean alignment"),
    ("L0", "loso", "shallow", False, False, "LOSO baseline: ShallowFBCSPNet, unseen subject"),
    ("L1", "loso", "shallow", True, False, "+ Euclidean alignment"),
    ("L2", "loso", "eegnet", True, False, "EEGNet + Euclidean alignment (vs. L1: model)"),
    ("L3", "loso", "shallow", True, True, "+ Euclidean alignment + augmentation (vs. L1: augmentation)"),
    ("L4", "loso", "conformer", True, False, "EEGConformer + Euclidean alignment (vs. L1: model)"),
]
BASELINE = {"within": "B0", "loso": "L0"}


def load(results_dir):
    rows = []
    for f in sorted(Path(results_dir).glob("*_seed*/subject_*.json")):
        r = json.loads(f.read_text())
        rows.append({k: r[k] for k in ("protocol", "model", "ea", "augment", "seed",
                                         "test_subject", "best_valid_epoch")}
                    | {"test_acc": 100 * r["test_acc"], "valid_acc": 100 * r["valid_acc"],
                       "test_last": 100 * r["test_acc_last_epoch"],
                       "train_acc": 100 * r["train_acc"], "train_time_s": r["train_time_s"]})
    return pd.DataFrame(rows)


def experiment_rows(df):
    out = []
    for eid, protocol, model, ea, aug, change in EXPERIMENTS:
        g = df[(df.protocol == protocol) & (df.model == model) & (df.ea == ea) & (df.augment == aug)]
        complete = g.groupby("seed").filter(lambda x: len(x) == 9)
        if complete.empty:
            continue
        per_seed = complete.groupby("seed")[["test_acc", "valid_acc", "train_acc", "test_last"]].mean()
        out.append({
            "id": eid, "protocol": protocol, "model": model, "EA": ea, "aug": aug, "change": change,
            "seeds": len(per_seed),
            "valid_mean": per_seed.valid_acc.mean(), "valid_std": per_seed.valid_acc.std(),
            "test_mean": per_seed.test_acc.mean(), "test_std": per_seed.test_acc.std(),
            "train_mean": per_seed.train_acc.mean(),
            "last_mean": per_seed.test_last.mean(), "last_std": per_seed.test_last.std(),
            "best_epoch_mean": complete.best_valid_epoch.mean(),
            "per_subject": complete.groupby("test_subject").test_acc.mean().round(2).to_dict(),
        })
    res = pd.DataFrame(out)
    if not res.empty:
        base = {p: res.loc[res.id == b, "test_mean"] for p, b in BASELINE.items()}
        res["delta_vs_baseline"] = [
            r.test_mean - base[r.protocol].iloc[0] if len(base[r.protocol]) else float("nan")
            for r in res.itertuples()]
    return res


def fmt(m, s):
    return f"{m:.2f}" if pd.isna(s) else f"{m:.2f} ± {s:.2f}"


def to_markdown(res):
    lines = ["| ID | Protocol | Change | Seeds | Valid acc (%) | Test acc (%), valid-selected | Δ vs. baseline (pp) | Test acc (%), after 100 epochs | Train acc (%) | Mean selected epoch |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res.itertuples():
        lines.append(f"| {r.id} | {r.protocol} | {r.change} | {r.seeds} | {fmt(r.valid_mean, r.valid_std)} "
                     f"| **{fmt(r.test_mean, r.test_std)}** | {r.delta_vs_baseline:+.2f} "
                     f"| {fmt(r.last_mean, r.last_std)} | {r.train_mean:.1f} | {r.best_epoch_mean:.0f} |")
    lines += ["", "Per-subject test accuracy (%), mean over seeds:", "",
              "| ID | " + " | ".join(str(s) for s in range(1, 10)) + " |",
              "|---|" + "---|" * 9]
    for r in res.itertuples():
        lines.append(f"| {r.id} | " + " | ".join(f"{r.per_subject.get(s, float('nan')):.1f}"
                                                 for s in range(1, 10)) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results_m2")
    args = parser.parse_args()
    df = load(args.results_dir)
    if df.empty:
        print("No result files found.")
        return
    res = experiment_rows(df)
    if res.empty:
        print("No experiment has all 9 subjects finished yet.")
        return
    res.drop(columns="per_subject").to_csv(Path(args.results_dir) / "summary.csv",
                                           index=False, float_format="%.2f")
    md = to_markdown(res)
    (Path(args.results_dir) / "summary.md").write_text(md, encoding="utf-8")
    print(md.encode("ascii", "replace").decode() if sys.stdout.encoding.lower() != "utf-8" else md)


if __name__ == "__main__":
    main()
