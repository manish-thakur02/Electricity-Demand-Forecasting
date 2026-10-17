# Validation completed on 10 October 2026

## Verified

- All six root notebooks executed top to bottom in fresh Jupyter kernels with 960 rows from the real CSV, smoke-test settings and no cell errors. Executed copies are in `validation/executed_notebooks/`.
- All six model paths also ran on the entire 43,848-row real CSV with normal model sizes and a 168-hour encoder. The neural runs were limited to **one complete training epoch**; Random Forest used 300 trees, and hybrid correctors used their normal settings.
- All models produced 6,552 finite, timestamp-aligned predictions for 3 April through 31 December 2024. Input fingerprints and evaluation protocols match.
- ARIMAX converged with Powell optimization on the full training period. Nonconverged full fits now raise an error.
- **21 regression tests passed**: causal demand/weather features, date/hour parsing, gaps and missing labels, split boundaries, timestamp joins, model/test isolation, neural future-value isolation, and dense-versus-packed LSTM outputs/states/gradients.
- Package consistency check reported no broken requirements. Exact local packages are recorded in `validation/environment-windows.txt`.
- Root notebooks contain no stale outputs. Executed evidence and current prediction/metric files are kept separately under `validation/`.

## Full-data execution checks

These durations include preparation/model execution on this machine while some checks ran concurrently. They are not controlled model-speed benchmarks.

| Model | Seconds | Test predictions | Execution status |
|---|---:|---:|---|
| random_forest | 21.6 | 6552 | Complete |
| arimax | 92.1 | 6552 | Converged |
| lstm | 106.5 | 6552 | Complete |
| lstm_rf | 156.2 | 6552 | Complete |
| tft | 277.9 | 6552 | Complete |
| tft_xgboost | 372.9 | 6552 | Complete |

The `validation/full_data_execution/` folder contains each model's predictions, metrics, plot and manifest. The comparison file checks consistency of the input and test periods before combining results.

## Interpretation and limits

**The one-epoch neural scores are execution-check results, not final trained-model rankings or a reproduction of the report.** Run the default longer training configurations to obtain corrected research results. Early stopping is enabled, and the best checkpoint is selected. Training until early stopping, robustness across seeds, and deployment forecasting are outside this execution check.

Verified environment: Windows, Python 3.12.14, CPU. The Colab/GPU setup instructions are provided, but GPU execution was not tested here.

Input audit: 43,848 raw rows; 43,680 after historical-feature warm-up; 11 missing hour labels recovered only after validating complete ordered 24-row days; 7 missing demand labels excluded from fitting/scoring; missing weather filled only from past observations. All 6,552 final test targets are observed.

The fixed-length LSTM optimization preserves architecture, initial weights and RNG state. A local representative backward-pass benchmark showed roughly a tenfold improvement over packed sequences; actual overall speed depends on model and hardware. Variable/zero-length inputs keep the upstream fallback and are covered by regression tests.

Third-party informational/deprecation warnings were retained; they were not hidden globally. Neural checkpoint/loading, training and prediction paths completed successfully.
