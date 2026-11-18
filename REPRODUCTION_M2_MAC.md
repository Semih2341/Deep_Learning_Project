# Milestone 2 – Independent reproduction on a second machine (CPU)

The M2 results in `results_m2/` were produced on a GPU laptop. To check that the main
within-subject finding does not depend on that setup, we re-ran the baseline (B0) and the
Euclidean alignment experiment (E4) on a second machine, using `src/improve.py` from `main`
unchanged, with the same seeds and the same data split. Results are in `results_m2_mac/`
(summarised by `python src/summarize_m2.py --results-dir results_m2_mac`).

## Environment

| | `results_m2/` | `results_m2_mac/` |
|---|---|---|
| Hardware | NVIDIA GeForce RTX 5050 Laptop GPU | Apple M1, CPU |
| OS | Windows 11 | macOS 15.3 |
| Python / PyTorch | 3.12.10 / 2.11.0+cu128 | 3.11.9 / 2.15.0 (CPU) |
| braindecode / MNE | 2.0.0 / 1.13.2 | 1.8.1 / 1.13.2 |

Commands (preprocessed arrays cached outside the repo; the split shapes matched
`results/data_splits.json`):

```bash
python src/prepare_data.py --subjects all --save-npz ../data_npz
python src/improve.py --protocol within --model shallow       --seed {20200220,1,2} --data-npz ../data_npz --out-dir results_m2_mac
python src/improve.py --protocol within --model shallow --ea  --seed {20200220,1,2} --data-npz ../data_npz --out-dir results_m2_mac
python src/summarize_m2.py --results-dir results_m2_mac
```

Training time was ≈ 3.5 min per subject on the CPU (≈ 3.5 h for all 54 runs).

## Results (trial-wise test accuracy on session `1test`, %, mean ± std over 3 seeds)

| | GPU (`results_m2/`) | CPU (`results_m2_mac/`) |
|---|---|---|
| B0, valid-selected | 60.58 ± 1.33 | 62.14 ± 1.64 |
| E4 (+ EA), valid-selected | 70.77 ± 1.12 | 69.47 ± 2.23 |
| **EA gain, valid-selected** | **+10.19** | **+7.33** |
| B0, after 100 epochs | 64.03 ± 0.10 | 64.44 ± 0.87 |
| E4 (+ EA), after 100 epochs | 71.48 ± 0.66 | 70.91 ± 1.29 |
| **EA gain, after 100 epochs** | **+7.45** | **+6.47** |

Per seed (valid-selected, B0 → E4):

| Seed | GPU | CPU |
|---|---|---|
| 20200220 | 59.92 → 70.99 | 63.73 → 70.64 |
| 1 | 59.72 → 69.56 | 60.46 → 66.90 |
| 2 | 62.11 → 71.76 | 62.23 → 70.87 |

Per subject (valid-selected, mean over seeds):

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| B0, GPU | 61.9 | 40.0 | 78.1 | 56.2 | 43.3 | 42.8 | 78.6 | 72.2 | 72.0 |
| B0, CPU | 65.3 | 43.1 | 79.9 | 57.6 | 47.2 | 43.4 | 78.1 | 73.7 | 71.0 |
| E4, GPU | 80.4 | 50.0 | 89.0 | 62.6 | 59.1 | 45.1 | 85.4 | 84.1 | 81.0 |
| E4, CPU | 79.9 | 51.4 | 87.7 | 63.5 | 54.5 | 43.6 | 83.3 | 80.9 | 80.3 |

## Observations

- **The EA effect reproduces.** EA improves the mean in every seed on both machines, and on the
  CPU it improves all 9 subjects (subject 6 only by +0.2 pp). The size of the gain differs
  (+7.3 vs +10.2 pp with valid-selected checkpoints).
- **Most of the difference comes from checkpoint selection, not from training.** After 100
  epochs the two machines agree within 0.6 pp for both B0 and E4, and the EA gain is
  +6.5 vs +7.5 pp. With the valid-selected checkpoint the differences are larger (up to
  1.6 pp in the mean and up to 12 pp for single subjects). For example, subject 1 with seed
  20200220 selects epoch 10 on the GPU (57.3 %) and epoch 60 on the CPU (69.4 %). Runs are
  not bit-identical across devices, and with only 48 validation trials a small change in the
  validation curve moves the selected epoch a lot. This supports the point in the M2
  discussion that the 48-trial validation set is a noisy selection criterion.
- The two runs also differ in braindecode version (2.0.0 vs 1.8.1), so this check also shows
  that `improve.py` gives the same results with both versions.
