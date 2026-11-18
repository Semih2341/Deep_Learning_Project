# Cross-Subject Motor Imagery Decoding on BCI Competition IV 2a

BME MSc Deep Learning – course project (Fall 2026/27).

4-class motor imagery decoding (left hand, right hand, feet, tongue) from EEG with
[braindecode](https://braindecode.org) and [MOABB](https://github.com/NeuroTechX/moabb).
The long-term goal is to study how a decoder can be made to work for new users with little
or no calibration data.

| Milestone | Content | Status |
|---|---|---|
| M1 – Reproduce | Cropped ShallowFBCSPNet tutorial, 100 epochs, all 9 subjects | done |
| M2 – Improve | EEGNet, EEGConformer, augmentation, Euclidean alignment, leave-one-subject-out | done |
| M3 – Contribute | Calibration trials, pretraining, channel / frequency analysis | – |

## Team

**Team name:** CetinxBen

| Name | Neptun code |
|---|---|
| Semih Çetin | E2NGJJ |
| Lovas Máté Benedek | CSY0C9 |

## Data

**Source.** BCI Competition IV, dataset 2a (Graz University of Technology; Brunner et al., 2008),
hosted on the [BNCI Horizon 2020 database](https://bnci-horizon-2020.eu/database/data-sets)
as *001-2014*. In MOABB it is called `BNCI2014_001`.

**Download.** Automatic. `braindecode.datasets.MOABBDataset` asks MOABB for a subject, and MOABB
downloads the original `.mat` files from the BNCI server on first use (~80 MB per subject). It
caches them in the MNE data folder (`--data-dir`, or Google Drive on Colab) and returns MNE `Raw`
objects. If the server is down, the download is retried up to 5 times.

**Content.** 9 subjects × 2 sessions recorded on different days (`0train`, `1test`). Each session
has 6 runs × 48 trials = 288 trials, balanced over 4 classes (left hand, right hand, feet, tongue;
72 each). 22 EEG + 3 EOG channels at 250 Hz, 4 s of motor imagery after each cue.

**Exploration** (`notebooks/00_data_exploration.ipynb`): class counts per subject and session,
electrode layout, raw trials, and per-class power spectra at C3/C4. The spectra show the expected
contralateral mu-band (8–13 Hz) decrease during hand imagery.

**Preparation** (`src/reproduce.py`, same as the braindecode tutorial): keep the 22 EEG channels →
V to µV → band-pass 4–38 Hz → exponential moving standardization → windows of 1000 samples from
0.5 s before the cue → cropped decoding (467 crop predictions per window).

**Train / validation / test** (`src/prepare_data.py`, within-subject, per subject):

| Split | Source | Trials | Inputs `X` | Outputs `y` |
|---|---|---|---|---|
| train | session `0train`, runs 0–4 | 240 | (480, 22, 1000) | (480,) class index 0–3 |
| valid | session `0train`, run 5 | 48 | (96, 22, 1000) | (96,) |
| test | session `1test`, runs 0–5 | 288 | (576, 22, 1000) | (576,) |

`X` is *(windows, channels, samples)*; each trial gives 2 windows. The validation run is held out
from training and is used for model selection from Milestone 2 on. The test session is used only
for the final evaluation. The M1 reproduction itself trains on the full `0train` session, as the
tutorial does. Shapes for all 9 subjects are in `results/data_splits.json`.

## Repository structure

```
├── notebooks/
│   ├── 00_data_exploration.ipynb  # data source, exploration plots, preparation, train/valid/test output
│   ├── 01_reproduce.ipynb         # training + results of Milestone 1 (Colab or local)
│   └── 02_improve.ipynb           # Milestone 2: results table, figures (saved with outputs)
├── src/
│   ├── prepare_data.py      # builds the train / valid / test sets, writes results/data_splits.json
│   ├── reproduce.py         # preprocessing, model, training + evaluation of one or more subjects
│   ├── summarize.py         # collects results into summary.csv / summary.md
│   ├── improve.py           # M2: EEGNet / EEGConformer / EA / augmentation, within-subject + LOSO
│   └── summarize_m2.py      # M2 results table → results_m2/summary.{md,csv}
├── results/
│   ├── <mode>_seed<seed>/subject_XX.json   # per-subject results, history, predictions
│   ├── logs/                # raw training logs
│   ├── data_splits.json     # shapes and class counts of the splits, all subjects
│   ├── summary.csv
│   └── summary.md
├── results_m2/              # M2: <protocol>_<model>[_ea][_aug]_seed<seed>/subject_XX.json, logs, summary
├── results_gpu/             # independent re-run of M1 on a second machine (GPU)
├── REPRODUCTION_GPU.md      # comparison of the GPU re-run with the original runs
├── results_m2_mac/          # M2 B0 and E4 re-run on a second machine (CPU)
├── REPRODUCTION_M2_MAC.md   # comparison of that re-run with results_m2/
├── run_m2.sh                # runs every M2 experiment
└── requirements.txt
```

## Setup

**Notebooks (run in this order):**
1. `notebooks/00_data_exploration.ipynb`: data source, exploration, preparation, train/valid/test sets.
2. `notebooks/01_reproduce.ipynb`: training and results.
3. `notebooks/02_improve.ipynb`: Milestone 2 results (see [Milestone 2 – Improve](#milestone-2--improve)).

**Colab / Kaggle:** open `notebooks/01_reproduce.ipynb`. By default it only loads the stored
results from `results/` and shows the tables and confusion matrices (no training). Set
`RERUN = True` to retrain everything on a GPU runtime; the notebook then keeps the pre-installed
PyTorch, installs only `braindecode` and `moabb`, and caches the dataset (BNCI2014_001,
~80 MB per subject, downloaded by MOABB) and new results on Google Drive.

**Local:**

```bash
pip install torch -r requirements.txt
python src/prepare_data.py --subjects all          # data: train/valid/test sets (optional: --save-npz data/)
python src/reproduce.py --subjects all --mode tutorial --epochs 100
python src/reproduce.py --subjects all --mode clean --epochs 100
python src/reproduce.py --subjects all --mode clean --epochs 100 --low-cut 0   # paper band 0-38 Hz
python src/summarize.py
```

Finished subjects are skipped on restart, so an interrupted run can be resumed with the same
command. If the BNCI data server is down, the download is retried automatically.

## Milestone 1 – Reproduce

### Pipeline

Identical to the braindecode example
[`plot_bcic_iv_2a_moabb_cropped.py`](https://braindecode.org/stable/auto_examples/model_building/plot_bcic_iv_2a_moabb_cropped.html);
the only change is `n_epochs = 100` (the example uses 4 to keep the documentation build short).

| Step | Setting |
|---|---|
| Data | BNCI2014_001 (BCIC IV 2a), 22 EEG channels, 250 Hz, 288 trials per session |
| Split | session `0train` → training, session `1test` → evaluation (within-subject) |
| Preprocessing | EEG channels only, V → µV, 4–38 Hz band-pass, exponential moving standardization (factor 1e-3, init block 1000) |
| Windows | trial start −0.5 s to trial end, 1000-sample input window, cropped training |
| Model | ShallowFBCSPNet, `final_conv_length=30`, converted to a dense prediction model |
| Training | AdamW, lr = 6.25e-4, weight decay 0, batch size 64, cosine annealing, `CroppedLoss` (cross-entropy on averaged crop predictions) |
| Metric | trial-wise accuracy: crop predictions are averaged within each trial |

### Two evaluation modes

The tutorial passes the test session to skorch as the validation split and uses
`EarlyStopping(patience=10, load_best=True)`. The reported checkpoint is therefore the epoch
with the lowest **test** loss, which is model selection on test data. To quantify this effect,
we run the pipeline in two modes:

- **`tutorial`** – exactly as in the example (reproduces the reported number).
- **`clean`** – no validation split and no early stopping: fixed 100 epochs, the final model is
  evaluated once on the test session.

All later milestones use only the clean protocol.

### Results

All numbers are trial-wise accuracy (%) on the held-out session `1test` (288 trials per subject,
chance level 25 %), seed 20200220, 4–38 Hz band as in the tutorial.

| Subject | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | **Mean ± std** |
|---|---|---|---|---|---|---|---|---|---|---|
| `tutorial` (early stopping on test) | 81.94 | 46.53 | 81.60 | 63.54 | 49.31 | 49.65 | 85.76 | 77.43 | 79.51 | **68.36 ± 16.13** |
| `clean` (fixed 100 epochs) | 80.56 | 43.06 | **84.38** | 66.67 | 59.72 | 54.17 | 86.46 | 76.04 | 78.47 | **69.95 ± 14.96** |
| Epoch selected in `tutorial` | 42 | 11 | 14 | 13 | 18 | 11 | 33 | 32 | 21 | |

Std is across the 9 subjects. Full tables (train accuracy, epochs, timing) are in
[`results/summary.md`](results/summary.md); per-subject histories and predictions are in
`results/<run>/subject_XX.json`.

**Seed variation (`clean`, 3 seeds: 20200220, 1, 2)**

| Subject | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | **Mean** |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean over seeds | 79.75 | 43.98 | **84.14** | 67.48 | 59.38 | 53.24 | 87.15 | 75.93 | 79.98 | **70.11** |
| Std over seeds | 1.12 | 0.87 | 0.40 | 1.40 | 0.35 | 1.06 | 0.69 | 0.20 | 1.31 | 0.16 |

The 9-subject mean is 70.25 / 70.14 / 69.95 for the three seeds, i.e. **70.11 ± 0.16 %**.
Run-to-run variation (≤ 1.4 pp per subject) is much smaller than the differences between
subjects, so the gap to the paper is not caused by an unlucky seed.

**Frequency band (`clean`, seed 20200220)**

| Band | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | **Mean ± std** |
|---|---|---|---|---|---|---|---|---|---|---|
| 4–38 Hz (tutorial) | 80.56 | 43.06 | 84.38 | 66.67 | 59.72 | 54.17 | 86.46 | 76.04 | 78.47 | **69.95 ± 14.96** |
| 0–38 Hz (paper) | 80.21 | 48.61 | 89.58 | 64.93 | 55.90 | 59.72 | 87.85 | 81.25 | 75.69 | **71.53 ± 14.71** |

Keeping the frequencies below 4 Hz helps subjects 2, 3, 6 and 8 (about +5 pp each) but hurts
5 and 9, so the value of the low-frequency content is subject-dependent.

**Comparison with the targets**

| | Ours (`clean`) | Ours (`tutorial`) | Reference |
|---|---|---|---|
| Subject 3 | 84.14 ± 0.40 (3 seeds) | 81.60 | 83.7 (braindecode tutorial) |
| Mean, 4–38 Hz | 70.11 ± 0.16 (3 seeds) | 68.36 | 71.9 (Schirrmeister et al., Table 2) |
| Mean, 0–38 Hz | 71.53 (1 seed) | – | 73.7 (Schirrmeister et al., Table 2) |

**Discussion**

- *Subject 3* is reproduced: 84.38 % with the clean protocol (+0.7 pp) and 81.60 % with the
  tutorial protocol (−2.1 pp), both within the expected 1–3 pp run-to-run variation.
- *Which paper number to compare to.* The 73.7 % of Schirrmeister et al. was obtained with a
  0–38 Hz band. The tutorial filters 4–38 Hz, for which the paper reports 71.9 % for the
  shallow ConvNet. Against this number the clean mean (70.11 % over 3 seeds) is 1.8 pp lower,
  which is within the normal range. Running the clean protocol with the paper's 0–38 Hz band
  (`--low-cut 0`) gives 71.53 % (seed 20200220): +1.6 pp over the same seed at 4–38 Hz and
  2.2 pp below the 73.7 % target. The paper reports +1.8 pp for the same change, so the effect
  of the band is reproduced as well.
- *Early stopping in the tutorial.* With `patience=10` on the noisy test loss, training stops
  after only 21–52 epochs. For subjects 2, 5 and 6 training accuracy is still at 79–83 % when it
  stops, i.e. the model is under-trained, and the cosine schedule (`T_max = 99`) has barely
  decayed the learning rate. The clean protocol trains all 100 epochs, never looks at the test
  session, and still gives a higher mean (+1.6 pp, largest gain on subject 5: +10.4 pp).
  Selecting the checkpoint on the test set therefore does not even help here; it only makes
  the reported number optimistically biased in principle.
- *Remaining differences to the paper.* The paper trains with its own early-stopping scheme on
  a validation split of the training session (Supporting Information A.4), not with a fixed
  number of epochs. Subject-level variance is large (43–86 %): subjects 2, 5 and 6 are known to be hard in BCIC IV 2a, while
  1, 3, 7, 8 and 9 are above 75 %.
- *Overfitting.* In the clean protocol, training accuracy reaches 100 % for every subject;
  the session-to-session gap (train 100 % vs. test 70 %) is the main limitation and the
  motivation for augmentation and alignment in Milestone 2.


### Environment

| | |
|---|---|
| Hardware | Apple M1, CPU only (macOS 15.3) |
| Software | Python 3.11.9, PyTorch 2.14.1, braindecode 1.8.1, MOABB 1.7.2, MNE 1.13.2, skorch 1.4.0 |
| Training time | ≈ 9 min per subject for 100 epochs (≈ 6 s / epoch); 81 min for all 9 subjects in `clean` mode (first run; ≈ 52 min in later runs) |

The versions are also stored in every result file (`env` field). The same code runs on a
Colab/Kaggle GPU through `notebooks/01_reproduce.ipynb`.


## Milestone 2 – Improve

All code for this milestone is in `src/improve.py` (training), `src/summarize_m2.py` (results
table), `run_m2.sh` (every experiment) and `notebooks/02_improve.ipynb` (table and figures, saved
with outputs).

### What we changed and why

Milestone 1 showed a large gap between the training session (100 % train accuracy) and the test
session recorded on another day (70 %). The project goal is a decoder for a *new* user. So we
tried changes that address overfitting and the shift between sessions and between people:

| Change | Why we expected it to help |
|---|---|
| **EEGNet** (Lawhern et al., 2018) | Compact depthwise/separable convolutions with ~3k parameters, against ~46k for ShallowFBCSPNet. We expected less overfitting on 240 training trials. |
| **EEGConformer** (Song et al., 2023) | Convolutional stem + self-attention, which can model longer-range temporal structure. Reported to beat ShallowFBCSPNet on BCIC IV 2a. |
| **Data augmentation** | Random `FrequencyShift` (±2 Hz), `SmoothTimeMask` (0.4 s) and `GaussianNoise`, each with p = 0.5, applied per training batch. More variety in the training data should reduce overfitting. `ChannelsSymmetry` is **not** used, because mirroring the hemispheres turns a left-hand trial into a right-hand one. |
| **Euclidean alignment (EA)** (He & Wu, 2019) | For each subject and session, the mean spatial covariance R of all trials is computed and each trial is whitened: X ← R^-½ X. This makes the second-order statistics of every session and subject the same, which directly targets the day-to-day and person-to-person shift. It needs no labels. |
| **Leave-one-subject-out (LOSO)** | This is not an improvement but the evaluation the project question asks for: train on 8 subjects, test on the 9th, who was never seen. |

### How we tested

- **Same data split for every run** (`src/prepare_data.py`). Within-subject: train = session
  `0train` runs 0–4, valid = run 5, test = session `1test`. LOSO fold *k*: train = runs 0–4 of
  the other 8 subjects (3840 windows), valid = run 5 of those 8 subjects, test = `1test` of
  subject *k*. The test trials are therefore the same in both protocols.
- **One factor at a time.** Each within-subject row (E1–E4) changes exactly one thing compared
  with the baseline B0. In LOSO, L1 adds EA to the baseline L0; L2, L3 and L4 each change one
  thing compared with L1.
- **Same budget and metric.** 100 epochs, batch 64, AdamW with cosine learning-rate decay. The
  learning rate is the default for each model (shallow 6.25e-4, EEGNet 1e-3, Conformer 2e-4).
  The metric is trial-wise accuracy, with the softmax averaged over the 2 windows of a trial.
- **Model selection on validation only.** After training, the weights of the epoch with the
  highest validation accuracy (ties broken by the lower validation loss) are restored, and the
  test set is evaluated **once**. As a second test-free rule, fixed in advance, we also report
  the weights after all 100 epochs (the M1 "clean" rule).
- **3 seeds** (20200220, 1, 2) for every row except the Conformer in LOSO (L4, 1 seed for
  compute reasons). The values are the mean over the 9 subjects (or folds), then mean ± std over
  the seeds.
- Note on EA: for the test session, R is computed from the unlabeled test trials of that
  session, as in the original EA method. No test labels are used anywhere.

### Results

| ID | Protocol | Change | Seeds | Valid acc (%) | **Test acc (%), valid-selected** | Δ vs. baseline (pp) | Test acc (%), after 100 epochs | Train acc (%) |
|---|---|---|---|---|---|---|---|---|
| M1 | within | *reference: M1 cropped ShallowFBCSPNet, trained on all 6 runs* | 3 | – | 70.11 ± 0.16 | – | 70.11 ± 0.16 | 100 |
| **B0** | within | **baseline:** ShallowFBCSPNet, trial-wise, 5 training runs | 3 | 68.67 ± 2.07 | **60.58 ± 1.33** | – | 64.03 ± 0.10 | 99.6 |
| E1 | within | model → EEGNet | 3 | 59.03 ± 0.93 | 59.61 ± 0.71 | −0.98 | 60.62 ± 0.49 | 91.1 |
| E2 | within | model → EEGConformer | 3 | 64.04 ± 0.48 | 58.81 ± 1.49 | −1.77 | 61.27 ± 1.05 | 86.0 |
| E3 | within | + data augmentation | 3 | 69.37 ± 1.04 | 65.05 ± 1.18 | +4.46 | 66.55 ± 1.08 | 99.8 |
| E4 | within | + Euclidean alignment | 3 | 70.99 ± 1.07 | **70.77 ± 1.12** | **+10.19** | 71.48 ± 0.66 | 100.0 |
| **L0** | LOSO | **baseline:** ShallowFBCSPNet, unseen subject | 3 | 63.78 ± 0.12 | **43.74 ± 0.56** | – | 43.49 ± 0.52 | 98.9 |
| L1 | LOSO | + Euclidean alignment | 3 | 68.28 ± 1.07 | 50.87 ± 0.69 | +7.14 | 51.38 ± 0.28 | 100.0 |
| L2 | LOSO | EEGNet + EA (vs. L1: model) | 3 | 65.63 ± 0.38 | 51.54 ± 1.36 | +7.81 | 52.47 ± 1.00 | 84.6 |
| L3 | LOSO | EA + augmentation (vs. L1: augmentation) | 3 | 69.94 ± 0.54 | **53.59 ± 0.95** | **+9.85** | 53.31 ± 1.18 | 99.9 |
| L4 | LOSO | EEGConformer + EA (vs. L1: model) | 1 | 66.61 | 50.93 | +7.19 | 50.42 | 98.1 |

Chance level is 25 %. Per subject (test accuracy, valid-selected, mean over seeds):

| ID | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| B0 | 61.9 | 40.0 | 78.1 | 56.2 | 43.3 | 42.8 | 78.6 | 72.2 | 72.0 |
| E1 | 67.5 | 43.8 | 81.2 | 52.9 | 33.2 | 36.7 | 70.4 | 71.6 | 79.2 |
| E2 | 65.9 | 37.5 | 78.6 | 59.6 | 37.3 | 40.0 | 72.7 | 69.7 | 68.1 |
| E3 | 71.6 | 41.5 | 84.5 | 64.0 | 46.4 | 46.4 | 80.0 | 74.8 | 76.2 |
| E4 | 80.4 | 50.0 | 89.0 | 62.6 | 59.1 | 45.1 | 85.4 | 84.1 | 81.0 |
| L0 | 65.7 | 30.2 | 58.3 | 35.9 | 27.8 | 31.6 | 32.2 | 56.0 | 55.9 |
| L1 | 68.6 | 33.0 | 68.2 | 46.3 | 35.6 | 44.4 | 39.5 | 62.7 | 59.5 |
| L2 | 67.8 | 33.1 | 72.2 | 40.7 | 36.8 | 43.9 | 44.0 | 66.1 | 59.3 |
| L3 | 72.9 | 37.6 | 69.4 | 47.6 | 37.4 | 44.6 | 43.4 | 67.5 | 61.9 |
| L4 | 69.1 | 29.9 | 69.1 | 44.4 | 36.5 | 44.8 | 39.6 | 63.9 | 61.1 |

The same tables are in [`results_m2/summary.md`](results_m2/summary.md). Every run is stored in
`results_m2/<protocol>_<model>[_ea][_aug]_seed<seed>/subject_XX.json` with its full validation
history and test predictions. The notebook also shows validation curves and confusion matrices.

**Second machine.** B0 and E4 were also run on a second machine (Apple M1 CPU, braindecode
1.8.1, same seeds and split; [`REPRODUCTION_M2_MAC.md`](REPRODUCTION_M2_MAC.md)). B0 gives
62.14 ± 1.64 % and E4 69.47 ± 2.23 %, so EA again improves every seed (+7.3 pp). After 100
epochs both machines agree within 0.6 pp. The larger differences between valid-selected
checkpoints come from the noisy 48-trial validation set (see Discussion).

### Discussion

- **Euclidean alignment helps most.** Within-subject it adds +10.2 pp (60.6 → 70.8 %) and
  improves every one of the 9 subjects, the hard subjects 5 (+16 pp) and 2 (+10 pp) included. With 5
  training runs it matches the M1 cropped model trained on all 6 runs. In LOSO it adds +7.1 pp.
  This fits our M1 analysis: the main problem is the shift in signal statistics between
  sessions and people, not the model's capacity.
- **Augmentation helps in both protocols.** It adds +4.5 pp within-subject and +2.7 pp on top
  of EA in LOSO (L3 vs. L1). L3 (EA + augmentation) is our best cross-subject model at
  53.6 %. Its cost is ~10× longer training in our implementation (augmentation runs per batch on the GPU).
- **The other models did not help.** EEGNet (E1) and EEGConformer (E2) are 1–2 pp *worse*
  than ShallowFBCSPNet within-subject. Their training accuracy (91 % and 86 %) shows they
  underfit within the 100-epoch budget on 480 windows. The EEGNet validation curve is still
  rising at epoch 100, so with a larger budget or a tuned learning rate it might catch up. With
  8× more data in LOSO they reach the level of ShallowFBCSPNet (L2 +0.7 pp, L4 +0.1 pp vs. L1),
  but the gain is within seed noise. The model architecture is not the bottleneck here.
- **Unseen subjects are much harder.** Even the best LOSO model (53.6 %) is 17 pp below the
  within-subject model with EA (70.8 %). LOSO validation accuracy (~68 %) is far above LOSO
  test accuracy (~51 %), because the validation trials come from the 8 training subjects, so
  validation does not measure transfer to a new person. Subjects 2, 5 and 7 stay below 45 % even
  with EA and augmentation. This is the motivation for Milestone 3 (a few calibration trials,
  fine-tuning).
- **Validation-based selection is noisy.** The validation set has only 48 trials per subject.
  Within-subject, the epoch it picks is often too early: B0 reaches 64.0 % after 100 epochs
  but only 60.6 % at the selected epoch. The conclusions (EA helps most, augmentation helps,
  the other models do not) are the same under both rules.
- **Why B0 is below M1 (60.6 vs. 70.1 %).** B0 trains on 5 runs instead of 6 (run 5 is held out
  for validation) and uses trial-wise instead of cropped training. We kept this setup fixed so
  that every row has a real validation set. The M1 row is only a reference.

### How to run

```bash
python src/prepare_data.py --subjects all --save-npz ../data_npz   # cache preprocessed arrays once
bash run_m2.sh python ../data_npz                                    # all experiments (~3 h GPU training on an RTX 5050 laptop)
python src/summarize_m2.py                                           # results_m2/summary.{md,csv}
```

Single experiments, for example:

```bash
python src/improve.py --protocol within --model shallow --ea --seed 1 --data-npz ../data_npz
python src/improve.py --protocol loso --model eegnet --ea --subjects 3 --data-npz ../data_npz
```

Finished subjects/folds are skipped, so an interrupted run can be resumed with the same command.
`notebooks/02_improve.ipynb` loads the stored results by default. Set `RERUN = True` to run
everything again.

**Environment (M2):** Windows 11, NVIDIA RTX 5050 Laptop GPU (8 GB), Python 3.12.10, PyTorch
2.11.0+cu128, **braindecode 2.0.0**, MNE 1.13.2, MOABB 1.7.2. M1 used braindecode 1.8.1
(`requirements.txt`), and `improve.py` works with both. The versions are stored in every result
file (`env` field). Total GPU training time for all runs was ≈ 3 h.


## References

- R. T. Schirrmeister et al., "Deep learning with convolutional neural networks for EEG decoding
  and visualization," *Human Brain Mapping*, 2017.
- V. J. Lawhern et al., "EEGNet: a compact convolutional neural network for EEG-based
  brain–computer interfaces," *J. Neural Eng.*, 2018.
- Y. Song et al., "EEG Conformer: Convolutional Transformer for EEG decoding and visualization,"
  *IEEE TNSRE*, 2023.
- H. He and D. Wu, "Transfer learning for brain–computer interfaces: a Euclidean space data
  alignment approach," *IEEE Trans. Biomed. Eng.*, 2019.
- braindecode: https://github.com/braindecode/braindecode
- MOABB: https://github.com/NeuroTechX/moabb
