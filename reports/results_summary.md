# Verified results

| Model | MAE | RMSE | R² | MAPE % |
|---|---:|---:|---:|---:|
| TFT+XGBoost | 150.835 | 205.268 | 0.978895 | 2.998 |
| TFT | 151.470 | 205.905 | 0.978764 | 3.004 |
| LSTM | 151.466 | 206.030 | 0.978738 | 3.017 |
| LSTM+Random Forest | 157.313 | 212.917 | 0.977293 | 3.104 |
| Random Forest | 170.233 | 228.483 | 0.973852 | 3.391 |
| ARIMAX | 278.319 | 365.013 | 0.933266 | 5.470 |

TFT+XGBoost has the lowest saved RMSE (205.268), 0.31% below standalone TFT. This single-seed difference does not establish statistical superiority. The upgraded TFT base differs from standalone TFT: its own RMSE was 207.765, reduced 1.20% by XGBoost. LSTM+Random Forest increased its own base RMSE from 206.030 to 212.917, a 3.34% deterioration. LSTM has marginally lower MAE than standalone TFT.

The optional weekly seasonal-naive reference has RMSE 330.171, MAE 243.479, R² 0.945398 and MAPE 4.809%. It is a reference benchmark, separate from the six trained models; ARIMAX performs worse than that reference.

The TFT+XGBoost export is a recovered evaluation using an existing epoch=30 checkpoint (zero-based) and refitted XGBoost. No new TFT training occurred during recovery. The configured maximum of 100 epochs does not establish 100 completed epochs, and the 59.371-second duration is recovery time only.
