# Milestone 1 – Independent reproduction on a second machine (GPU)

This branch re-runs the Milestone 1 pipeline from `main` unchanged (`src/reproduce.py`,
same seeds, same CLI) on different hardware, operating system and PyTorch version. It also
adds two more seeds for the 0–38 Hz band, which has only one seed on `main`. Results are in
`results_gpu/`, and that folder is summarised by `python src/summarize.py --results-dir results_gpu`.

## Environment

| | `main` (original runs) | This branch |
|---|---|---|
| Hardware | Apple M1, CPU | NVIDIA GeForce RTX 5050 Laptop GPU |
| OS | macOS 15.3 | Windows 11 |
| Python / PyTorch | 3.11.9 / 2.14.1 (CPU) | 3.12.10 / 2.11.0+cu128 |
| braindecode / MOABB / MNE / skorch | 1.8.1 / 1.7.2 / 1.13.2 / 1.4.0 | same |

Commands (data cached outside the repo):

```bash
python src/reproduce.py --subjects all --mode tutorial --epochs 100 --out-dir results_gpu --data-dir ../mne_data
python src/reproduce.py --subjects all --mode clean    --epochs 100 --out-dir results_gpu --data-dir ../mne_data --seed {20200220,1,2}
python src/reproduce.py --subjects all --mode clean    --epochs 100 --out-dir results_gpu --data-dir ../mne_data --seed {20200220,1,2} --low-cut 0
python src/summarize.py --results-dir results_gpu
```

## Results (trial-wise test accuracy on session `1test`, %)

| Run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| `tutorial`, seed 20200220 | 80.90 | 38.89 | 84.03 | 64.58 | 50.35 | 43.06 | 88.19 | 73.61 | 77.78 | 66.82 |
| `clean`, seed 20200220 | 80.21 | 43.40 | 84.72 | 69.44 | 57.64 | 53.47 | 87.50 | 76.04 | 78.12 | 70.06 |
| `clean`, seed 1 | 77.78 | 48.26 | 84.72 | 64.93 | 58.33 | 51.74 | 88.19 | 77.08 | 77.43 | 69.83 |
| `clean`, seed 2 | 77.78 | 43.75 | 85.07 | 70.14 | 59.03 | 49.65 | 87.50 | 77.43 | 79.86 | 70.02 |
| `clean` 0–38 Hz, seed 20200220 | 80.56 | 46.53 | 88.19 | 65.28 | 56.60 | 58.33 | 86.11 | 80.21 | 77.43 | 71.03 |
| `clean` 0–38 Hz, seed 1 | 78.12 | 48.96 | 89.24 | 67.71 | 61.81 | 55.90 | 84.38 | 81.25 | 78.82 | 71.80 |
| `clean` 0–38 Hz, seed 2 | 77.78 | 43.40 | 88.19 | 66.32 | 59.38 | 56.94 | 84.03 | 78.12 | 78.82 | 70.33 |

## Comparison with `main` and the targets

| | This branch (GPU) | `main` (CPU) | Reference |
|---|---|---|---|
| Subject 3, `clean` 4–38 Hz | 84.84 ± 0.20 (3 seeds) | 84.14 ± 0.40 (3 seeds) | 83.7 (tutorial) |
| Subject 3, `tutorial` | 84.03 | 81.60 | 83.7 (tutorial) |
| Mean, `clean` 4–38 Hz | **69.97 ± 0.12** (3 seeds) | 70.11 ± 0.16 (3 seeds) | 71.9 (paper, 4–38 Hz) |
| Mean, `clean` 0–38 Hz | **71.05 ± 0.73** (3 seeds) | 71.53 (1 seed) | 73.7 (paper, 0–38 Hz) |
| Mean, `tutorial` | 66.82 | 68.36 | – |

## Observations

- **The clean protocol reproduces across machines.** On a different OS, PyTorch version and
  device (CUDA instead of CPU), the 3-seed mean differs from `main` by only 0.14 pp
  (69.97 vs 70.11). The difference is about the size of the seed std, so the M1 numbers do not
  depend on the setup. Results for a given seed are not bit-identical across devices, because
  the order of floating-point operations differs between CPU and cuDNN kernels.
- **The tutorial protocol is less stable.** The same seed gives 66.82 here and 68.36 on `main`
  (−1.5 pp). The early-stopping epoch, selected on the noisy test loss, changes between machines:
  subject 2 stopped at epoch 8 (38.89 %) and subject 6 at epoch 9 (43.06 %), and both are
  under-trained. This supports the decision on `main` to use the clean protocol: it is both
  test-free and more reproducible.
- **The 0–38 Hz effect is confirmed with 3 seeds.** Keeping the < 4 Hz content raises the mean
  by +1.08 pp (71.05 vs 69.97). The paper reports +1.8 pp, and the single seed on `main` gave
  +1.6 pp. Subject 3 benefits most (+3.7 pp, 88.54 ± 0.60). The gap to the paper's 73.7 %
  (−2.65 pp) stays at the edge of the 1–3 pp range that the course considers normal.
- **Run time.** On the GPU, one subject trains for 100 epochs in 10–20 s when the laptop is
  not throttled. Data loading and preprocessing take longer than training. Wall-clock times in
  the JSON files vary (up to ~5 min per subject) with the laptop's power state.
