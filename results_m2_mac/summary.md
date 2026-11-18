| ID | Protocol | Change | Seeds | Valid acc (%) | Test acc (%), valid-selected | Δ vs. baseline (pp) | Test acc (%), after 100 epochs | Train acc (%) | Mean selected epoch |
|---|---|---|---|---|---|---|---|---|---|
| B0 | within | baseline: ShallowFBCSPNet (M1 model, trial-wise, valid-selected) | 3 | 68.98 ± 0.40 | **62.14 ± 1.64** | +0.00 | 64.44 ± 0.87 | 99.8 | 52 |
| E4 | within | + Euclidean alignment | 3 | 70.99 ± 2.25 | **69.47 ± 2.23** | +7.33 | 70.91 ± 1.29 | 100.0 | 42 |

Per-subject test accuracy (%), mean over seeds:

| ID | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| B0 | 65.3 | 43.1 | 79.9 | 57.6 | 47.2 | 43.4 | 78.1 | 73.7 | 71.0 |
| E4 | 79.9 | 51.4 | 87.7 | 63.5 | 54.5 | 43.6 | 83.3 | 80.9 | 80.3 |
