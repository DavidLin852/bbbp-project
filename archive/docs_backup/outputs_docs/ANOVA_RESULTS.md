# ANOVA Analysis Results

Generated: 2026-02-06 11:36:58

================================================================================

## One-way ANOVA Results

### By Feature and Metric

| Feature | Metric | F-statistic | p-value | Significant |
|--------|--------|-------------|---------|-------------|
| morgan | auc | 0.0739 | 0.0000e+00 | Yes |
| morgan | f1 | 0.0343 | 0.0000e+00 | No |
| morgan | mcc | 0.1568 | 0.0000e+00 | Yes |
| maccs | auc | 0.1077 | 0.0000e+00 | Yes |
| maccs | f1 | 0.0871 | 0.0000e+00 | Yes |
| maccs | mcc | 0.3025 | 0.0000e+00 | Yes |
| atompairs | auc | 0.2519 | 0.0000e+00 | Yes |
| atompairs | f1 | 0.1301 | 0.0000e+00 | Yes |
| atompairs | mcc | 0.4428 | 0.0000e+00 | Yes |
| fp2 | auc | 0.1841 | 0.0000e+00 | Yes |
| fp2 | f1 | 0.1201 | 0.0000e+00 | Yes |
| fp2 | mcc | 0.3574 | 0.0000e+00 | Yes |
| rdkit_desc | auc | 0.1878 | 0.0000e+00 | Yes |
| rdkit_desc | f1 | 0.1670 | 0.0000e+00 | Yes |
| rdkit_desc | mcc | 0.4394 | 0.0000e+00 | Yes |

## Two-way ANOVA Results (Model × Feature Interaction)

### Model Effects (deviation from grand mean)

| Model | Effect |
|-------|--------|

**AUC**
| RF | +0.0359 |
| XGB | +0.0332 |
| LGBM | +0.0311 |
| GB | +0.0300 |
| SVM_RBF | +0.0192 |
| ETC | +0.0142 |
| KNN5 | -0.0106 |
| ADA | -0.0118 |
| LR | -0.0204 |
| NB_Bernoulli | -0.1206 |

**F1**
| LGBM | +0.0190 |
| GB | +0.0180 |
| KNN5 | +0.0174 |
| XGB | +0.0169 |
| ETC | +0.0153 |
| RF | +0.0140 |
| SVM_RBF | +0.0113 |
| ADA | -0.0098 |
| LR | -0.0159 |
| NB_Bernoulli | -0.0862 |

**MCC**
| LGBM | +0.0662 |
| GB | +0.0597 |
| KNN5 | +0.0578 |
| XGB | +0.0566 |
| ETC | +0.0482 |
| RF | +0.0422 |
| SVM_RBF | +0.0383 |
| LR | -0.0425 |
| ADA | -0.0640 |
| NB_Bernoulli | -0.2625 |

### Feature Effects

| Feature | Effect |
|--------|--------|

**AUC**
| morgan | +0.0114 |
| maccs | +0.0063 |
| fp2 | -0.0027 |
| rdkit_desc | -0.0052 |
| atompairs | -0.0098 |

**F1**
| morgan | +0.0104 |
| maccs | +0.0047 |
| fp2 | +0.0028 |
| atompairs | -0.0050 |
| rdkit_desc | -0.0129 |

**MCC**
| morgan | +0.0274 |
| maccs | +0.0167 |
| fp2 | +0.0136 |
| atompairs | -0.0208 |
| rdkit_desc | -0.0369 |

## Pairwise Comparisons (Effect Size Analysis)

Significant pairs (Cohen's d > 0.8):

### morgan

| Group 1 | Group 2 | Difference | Cohen's d |
|---------|---------|------------|-----------|
| RF | KNN5 | +0.0739 | 7.39 |
| XGB | KNN5 | +0.0720 | 7.20 |
| GB | KNN5 | +0.0697 | 6.97 |
| LGBM | KNN5 | +0.0672 | 6.72 |
| RF | ADA | +0.0596 | 5.96 |
| XGB | ADA | +0.0577 | 5.77 |
| GB | ADA | +0.0554 | 5.54 |
| SVM_RBF | KNN5 | +0.0535 | 5.35 |
| RF | NB_Bernoulli | +0.0532 | 5.32 |
| LGBM | ADA | +0.0529 | 5.29 |
| XGB | NB_Bernoulli | +0.0514 | 5.14 |
| ETC | KNN5 | +0.0504 | 5.04 |
| GB | NB_Bernoulli | +0.0491 | 4.91 |
| LGBM | NB_Bernoulli | +0.0466 | 4.66 |
| RF | LR | +0.0442 | 4.42 |
| XGB | LR | +0.0423 | 4.23 |
| GB | LR | +0.0400 | 4.00 |
| SVM_RBF | ADA | +0.0393 | 3.93 |
| LGBM | LR | +0.0375 | 3.75 |
| ETC | ADA | +0.0361 | 3.61 |
| SVM_RBF | NB_Bernoulli | +0.0329 | 3.29 |
| ETC | NB_Bernoulli | +0.0297 | 2.97 |
| LR | KNN5 | +0.0297 | 2.97 |
| SVM_RBF | LR | +0.0239 | 2.39 |
| RF | ETC | +0.0235 | 2.35 |
| XGB | ETC | +0.0216 | 2.16 |
| ETC | LR | +0.0207 | 2.07 |
| NB_Bernoulli | KNN5 | +0.0207 | 2.07 |
| RF | SVM_RBF | +0.0203 | 2.03 |
| GB | ETC | +0.0193 | 1.93 |
| XGB | SVM_RBF | +0.0185 | 1.85 |
| LGBM | ETC | +0.0168 | 1.68 |
| GB | SVM_RBF | +0.0162 | 1.62 |
| LR | ADA | +0.0154 | 1.54 |
| ADA | KNN5 | +0.0143 | 1.43 |
| LGBM | SVM_RBF | +0.0137 | 1.37 |
| LR | NB_Bernoulli | +0.0090 | 0.90 |

### maccs

| Group 1 | Group 2 | Difference | Cohen's d |
|---------|---------|------------|-----------|
| RF | NB_Bernoulli | +0.1077 | 10.77 |
| XGB | NB_Bernoulli | +0.1019 | 10.19 |
| LGBM | NB_Bernoulli | +0.1002 | 10.02 |
| GB | NB_Bernoulli | +0.0956 | 9.56 |
| SVM_RBF | NB_Bernoulli | +0.0914 | 9.14 |
| ETC | NB_Bernoulli | +0.0845 | 8.45 |
| KNN5 | NB_Bernoulli | +0.0684 | 6.84 |
| RF | ADA | +0.0582 | 5.82 |
| RF | LR | +0.0581 | 5.81 |
| XGB | ADA | +0.0524 | 5.24 |
| XGB | LR | +0.0523 | 5.23 |
| LGBM | ADA | +0.0507 | 5.07 |
| LGBM | LR | +0.0505 | 5.05 |
| LR | NB_Bernoulli | +0.0497 | 4.97 |
| ADA | NB_Bernoulli | +0.0495 | 4.95 |
| GB | ADA | +0.0461 | 4.61 |
| GB | LR | +0.0460 | 4.60 |
| SVM_RBF | ADA | +0.0419 | 4.19 |
| SVM_RBF | LR | +0.0417 | 4.17 |
| RF | KNN5 | +0.0393 | 3.93 |
| ETC | ADA | +0.0350 | 3.50 |
| ETC | LR | +0.0349 | 3.49 |
| XGB | KNN5 | +0.0335 | 3.35 |
| LGBM | KNN5 | +0.0318 | 3.18 |
| GB | KNN5 | +0.0272 | 2.72 |
| RF | ETC | +0.0232 | 2.32 |
| SVM_RBF | KNN5 | +0.0230 | 2.30 |
| KNN5 | ADA | +0.0189 | 1.89 |
| KNN5 | LR | +0.0187 | 1.87 |
| XGB | ETC | +0.0174 | 1.74 |
| RF | SVM_RBF | +0.0163 | 1.63 |
| ETC | KNN5 | +0.0161 | 1.61 |
| LGBM | ETC | +0.0157 | 1.57 |
| RF | GB | +0.0121 | 1.21 |
| GB | ETC | +0.0111 | 1.11 |
| XGB | SVM_RBF | +0.0105 | 1.05 |
| LGBM | SVM_RBF | +0.0088 | 0.88 |

### atompairs

| Group 1 | Group 2 | Difference | Cohen's d |
|---------|---------|------------|-----------|
| XGB | NB_Bernoulli | +0.2519 | 25.19 |
| LGBM | NB_Bernoulli | +0.2503 | 25.03 |
| RF | NB_Bernoulli | +0.2495 | 24.95 |
| GB | NB_Bernoulli | +0.2464 | 24.64 |
| SVM_RBF | NB_Bernoulli | +0.2394 | 23.94 |
| ETC | NB_Bernoulli | +0.2281 | 22.81 |
| KNN5 | NB_Bernoulli | +0.2104 | 21.04 |
| ADA | NB_Bernoulli | +0.2101 | 21.01 |
| LR | NB_Bernoulli | +0.1668 | 16.68 |
| XGB | LR | +0.0852 | 8.52 |
| LGBM | LR | +0.0835 | 8.35 |
| RF | LR | +0.0827 | 8.27 |
| GB | LR | +0.0796 | 7.96 |
| SVM_RBF | LR | +0.0726 | 7.26 |
| ETC | LR | +0.0613 | 6.13 |
| KNN5 | LR | +0.0436 | 4.36 |
| ADA | LR | +0.0433 | 4.33 |
| XGB | ADA | +0.0418 | 4.18 |
| XGB | KNN5 | +0.0415 | 4.15 |
| LGBM | ADA | +0.0402 | 4.02 |
| LGBM | KNN5 | +0.0399 | 3.99 |
| RF | ADA | +0.0394 | 3.94 |
| RF | KNN5 | +0.0391 | 3.91 |
| GB | ADA | +0.0363 | 3.63 |
| GB | KNN5 | +0.0360 | 3.60 |
| SVM_RBF | ADA | +0.0293 | 2.93 |
| SVM_RBF | KNN5 | +0.0290 | 2.90 |
| XGB | ETC | +0.0239 | 2.39 |
| LGBM | ETC | +0.0222 | 2.22 |
| RF | ETC | +0.0214 | 2.14 |
| GB | ETC | +0.0183 | 1.83 |
| ETC | ADA | +0.0180 | 1.80 |
| ETC | KNN5 | +0.0177 | 1.77 |
| XGB | SVM_RBF | +0.0125 | 1.25 |
| SVM_RBF | ETC | +0.0113 | 1.13 |
| LGBM | SVM_RBF | +0.0109 | 1.09 |
| RF | SVM_RBF | +0.0101 | 1.01 |

### fp2

| Group 1 | Group 2 | Difference | Cohen's d |
|---------|---------|------------|-----------|
| RF | NB_Bernoulli | +0.1841 | 18.41 |
| XGB | NB_Bernoulli | +0.1785 | 17.85 |
| LGBM | NB_Bernoulli | +0.1780 | 17.80 |
| GB | NB_Bernoulli | +0.1775 | 17.75 |
| SVM_RBF | NB_Bernoulli | +0.1773 | 17.73 |
| ETC | NB_Bernoulli | +0.1656 | 16.56 |
| ADA | NB_Bernoulli | +0.1612 | 16.12 |
| KNN5 | NB_Bernoulli | +0.1480 | 14.80 |
| LR | NB_Bernoulli | +0.1436 | 14.36 |
| RF | LR | +0.0405 | 4.05 |
| RF | KNN5 | +0.0362 | 3.62 |
| XGB | LR | +0.0350 | 3.50 |
| LGBM | LR | +0.0345 | 3.45 |
| GB | LR | +0.0339 | 3.39 |
| SVM_RBF | LR | +0.0338 | 3.38 |
| XGB | KNN5 | +0.0306 | 3.06 |
| LGBM | KNN5 | +0.0301 | 3.01 |
| GB | KNN5 | +0.0295 | 2.95 |
| SVM_RBF | KNN5 | +0.0294 | 2.94 |
| RF | ADA | +0.0229 | 2.29 |
| ETC | LR | +0.0221 | 2.21 |
| RF | ETC | +0.0185 | 1.85 |
| ETC | KNN5 | +0.0177 | 1.77 |
| ADA | LR | +0.0177 | 1.77 |
| XGB | ADA | +0.0173 | 1.73 |
| LGBM | ADA | +0.0168 | 1.68 |
| GB | ADA | +0.0163 | 1.63 |
| SVM_RBF | ADA | +0.0161 | 1.61 |
| ADA | KNN5 | +0.0133 | 1.33 |
| XGB | ETC | +0.0129 | 1.29 |
| LGBM | ETC | +0.0124 | 1.24 |
| GB | ETC | +0.0119 | 1.19 |
| SVM_RBF | ETC | +0.0117 | 1.17 |

### rdkit_desc

| Group 1 | Group 2 | Difference | Cohen's d |
|---------|---------|------------|-----------|
| RF | NB_Bernoulli | +0.1878 | 18.78 |
| XGB | NB_Bernoulli | +0.1850 | 18.50 |
| GB | NB_Bernoulli | +0.1844 | 18.44 |
| LGBM | NB_Bernoulli | +0.1833 | 18.33 |
| ETC | NB_Bernoulli | +0.1659 | 16.59 |
| SVM_RBF | NB_Bernoulli | +0.1580 | 15.80 |
| KNN5 | NB_Bernoulli | +0.1436 | 14.36 |
| LR | NB_Bernoulli | +0.1318 | 13.18 |
| ADA | NB_Bernoulli | +0.1292 | 12.92 |
| RF | ADA | +0.0586 | 5.86 |
| RF | LR | +0.0560 | 5.60 |
| XGB | ADA | +0.0557 | 5.57 |
| GB | ADA | +0.0551 | 5.51 |
| LGBM | ADA | +0.0540 | 5.40 |
| XGB | LR | +0.0532 | 5.32 |
| GB | LR | +0.0526 | 5.26 |
| LGBM | LR | +0.0515 | 5.15 |
| RF | KNN5 | +0.0442 | 4.42 |
| XGB | KNN5 | +0.0413 | 4.13 |
| GB | KNN5 | +0.0407 | 4.07 |
| LGBM | KNN5 | +0.0396 | 3.96 |
| ETC | ADA | +0.0366 | 3.66 |
| ETC | LR | +0.0341 | 3.41 |
| RF | SVM_RBF | +0.0298 | 2.98 |
| SVM_RBF | ADA | +0.0287 | 2.87 |
| XGB | SVM_RBF | +0.0270 | 2.70 |
| GB | SVM_RBF | +0.0264 | 2.64 |
| SVM_RBF | LR | +0.0262 | 2.62 |
| LGBM | SVM_RBF | +0.0253 | 2.53 |
| ETC | KNN5 | +0.0222 | 2.22 |
| RF | ETC | +0.0219 | 2.19 |
| XGB | ETC | +0.0191 | 1.91 |
| GB | ETC | +0.0185 | 1.85 |
| LGBM | ETC | +0.0174 | 1.74 |
| KNN5 | ADA | +0.0144 | 1.44 |
| SVM_RBF | KNN5 | +0.0144 | 1.44 |
| KNN5 | LR | +0.0118 | 1.18 |

## Statistical Summary

### Overall Statistics

- Total observations: 57
- Grand mean AUC: 0.9350
- AUC std: 0.0488
- AUC range: 0.7170 - 0.9724

### Model Rankings (by Mean AUC)

| Rank | Model | Mean AUC | Std | Count |
|------|-------|----------|-----|-------|
| 1 | RF | 0.9684 | 0.0037 | 6 |
| 2 | XGB | 0.9659 | 0.0050 | 6 |
| 3 | LGBM | 0.9640 | 0.0043 | 6 |
| 4 | GB | 0.9626 | 0.0045 | 6 |
| 5 | SVM_RBF | 0.9514 | 0.0076 | 5 |
| 6 | ETC | 0.9467 | 0.0020 | 6 |
| 7 | ADA | 0.9223 | 0.0122 | 6 |
| 8 | KNN5 | 0.9215 | 0.0132 | 5 |
| 9 | LR | 0.9151 | 0.0172 | 6 |
| 10 | NB_Bernoulli | 0.8116 | 0.0796 | 5 |