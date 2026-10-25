"""Collect per-subject result files into a CSV and a Markdown table.

Usage:
    python src/summarize.py --results-dir results
"""

import argparse
import json
from pathlib import Path

import pandas as pd

TARGET_SUBJECT_3 = 83.7  # braindecode tutorial, 100 epochs
# Schirrmeister et al. 2017, Table 2, ShallowConvNet on BCIC IV 2a
PAPER_MEAN = {0.0: 73.7, 4.0: 71.9}  # by high-pass cut-off (0-38 Hz / 4-38 Hz)


def load_runs(results_dir):
    rows = []
    for f in sorted(Path(results_dir).glob("*_seed*/subject_*.json")):
        r = json.loads(f.read_text())
        rows.append(
            {
                "mode": r["mode"],
                "seed": r["seed"],
                "low_cut_hz": r.get("low_cut_hz", 4.0),
                "subject": r["subject"],
                "test_acc": 100 * r["test_acc"],
                "train_acc": 100 * r["train_acc"],
                "selected_epoch": r["selected_epoch"],
                "epochs_run": r["n_epochs_run"],
                "train_time_s": r["train_time_s"],
            }
        )
    return pd.DataFrame(rows)


def seed_summary(df):
    """Mean over subjects for each seed, then mean +- std over seeds."""
    lines = [
        "### Across seeds\n",
        "| Mode | Band | Seeds | Mean acc (%) over seeds | Subject 3 (%) |",
        "|---|---|---|---|---|",
    ]
    for (mode, low_cut), g in df.groupby(["mode", "low_cut_hz"]):
        per_seed = g.groupby("seed").filter(lambda x: len(x) == 9).groupby("seed")
        means = per_seed.test_acc.mean()
        s3 = g[g.subject == 3].test_acc
        if means.empty:
            continue
        std = f" ± {means.std():.2f}" if len(means) > 1 else ""
        s3_std = f" ± {s3.std():.2f}" if len(s3) > 1 else ""
        lines.append(
            f"| {mode} | {low_cut:g}-38 Hz | {len(means)} | {means.mean():.2f}{std} "
            f"| {s3.mean():.2f}{s3_std} |"
        )
    lines.append("")
    return lines


def to_markdown(df):
    lines = []
    for (mode, low_cut, seed), g in df.groupby(["mode", "low_cut_hz", "seed"]):
        g = g.sort_values("subject")
        lines.append(f"### mode = `{mode}`, band = {low_cut:g}-38 Hz, seed = {seed}\n")
        lines.append("| Subject | Test acc (%) | Train acc (%) | Selected epoch | Epochs run |")
        lines.append("|---|---|---|---|---|")
        for _, r in g.iterrows():
            lines.append(
                f"| {r.subject} | {r.test_acc:.2f} | {r.train_acc:.2f} "
                f"| {r.selected_epoch} | {r.epochs_run} |"
            )
        lines.append(
            f"| **Mean ± std** | **{g.test_acc.mean():.2f} ± {g.test_acc.std():.2f}** "
            f"| {g.train_acc.mean():.2f} | | |"
        )
        lines.append("")
        if (g.subject == 3).any():
            s3 = g.loc[g.subject == 3, "test_acc"].item()
            lines.append(f"Subject 3: {s3:.2f}% (target {TARGET_SUBJECT_3}%)  ")
        if len(g) == 9:
            lines.append(
                f"Mean over 9 subjects: {g.test_acc.mean():.2f}% "
                f"(paper, same band: {PAPER_MEAN.get(low_cut, 'n/a')}%)"
            )
        lines.append("")
    lines += seed_summary(df)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    df = load_runs(args.results_dir)
    if df.empty:
        print("No result files found.")
        return
    df.sort_values(["mode", "low_cut_hz", "seed", "subject"]).to_csv(
        Path(args.results_dir) / "summary.csv", index=False, float_format="%.2f"
    )
    md = to_markdown(df)
    (Path(args.results_dir) / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
