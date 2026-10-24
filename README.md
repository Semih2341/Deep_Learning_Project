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

**Colab / Kaggle:** open `notebooks/01_reproduce.ipynb`, select a GPU runtime and run all cells.
The notebook keeps the pre-installed PyTorch and installs only `braindecode` and `moabb`.
The dataset (BNCI2014_001, ~80 MB per subject) is downloaded automatically by MOABB and
cached on Google Drive.

**Local:**

```bash
pip install torch -r requirements.txt
python src/reproduce.py --subjects all --mode tutorial --epochs 100
python src/reproduce.py --subjects all --mode clean --epochs 100
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

<!-- RESULTS -->

### Environment

<!-- ENV -->

## References

- R. T. Schirrmeister et al., "Deep learning with convolutional neural networks for EEG decoding
  and visualization," *Human Brain Mapping*, 2017.
- V. J. Lawhern et al., "EEGNet: a compact convolutional neural network for EEG-based
  brain–computer interfaces," *J. Neural Eng.*, 2018.
- Y. Song et al., "EEG Conformer: Convolutional Transformer for EEG decoding and visualization,"
  *IEEE TNSRE*, 2023.
- braindecode: https://github.com/braindecode/braindecode
- MOABB: https://github.com/NeuroTechX/moabb
