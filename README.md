# Cross-Subject Motor Imagery Decoding on BCI Competition IV 2a

BME MSc Deep Learning – course project (Fall 2026/27).

4-class motor imagery decoding (left hand, right hand, feet, tongue) from EEG with
[braindecode](https://braindecode.org) and [MOABB](https://github.com/NeuroTechX/moabb).
The long-term goal is to study how a decoder can be made to work for new users with little
or no calibration data.

| Milestone | Content | Status |
|---|---|---|
| M1 – Reproduce | Cropped ShallowFBCSPNet tutorial, 100 epochs, all 9 subjects | done |
| M2 – Improve | EEGNet, EEGConformer, augmentation, Euclidean alignment, leave-one-subject-out | – |
| M3 – Contribute | Calibration trials, pretraining, channel / frequency analysis | – |

## Repository structure

```
├── notebooks/
│   └── 01_reproduce.ipynb   # Colab notebook for Milestone 1
├── src/
│   ├── reproduce.py         # training + evaluation of one or more subjects
│   └── summarize.py         # collects results into summary.csv / summary.md
├── results/
│   ├── <mode>_seed<seed>/subject_XX.json   # per-subject results, history, predictions
│   ├── logs/                # raw training logs
│   ├── summary.csv
│   └── summary.md
└── requirements.txt
```

## Setup

**Colab / Kaggle:** open `notebooks/01_reproduce.ipynb`. By default it only loads the stored
results from `results/` and shows the tables and confusion matrices (no training). Set
`RERUN = True` to retrain everything on a GPU runtime; the notebook then keeps the pre-installed
PyTorch, installs only `braindecode` and `moabb`, and caches the dataset (BNCI2014_001,
~80 MB per subject, downloaded by MOABB) and new results on Google Drive.

**Local:**

```bash
pip install torch -r requirements.txt
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


## References

- R. T. Schirrmeister et al., "Deep learning with convolutional neural networks for EEG decoding
  and visualization," *Human Brain Mapping*, 2017.
- V. J. Lawhern et al., "EEGNet: a compact convolutional neural network for EEG-based
  brain–computer interfaces," *J. Neural Eng.*, 2018.
- Y. Song et al., "EEG Conformer: Convolutional Transformer for EEG decoding and visualization,"
  *IEEE TNSRE*, 2023.
- braindecode: https://github.com/braindecode/braindecode
- MOABB: https://github.com/NeuroTechX/moabb
