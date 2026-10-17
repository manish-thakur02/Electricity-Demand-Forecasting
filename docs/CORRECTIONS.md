# Corrections and experiment cleanup

## Shared corrections

1. Combine date and separate hour, validate hourly continuity, reject duplicate or
   ambiguous timestamps, and reconstruct calendar fields instead of trusting stale columns.
2. Shift rolling statistics, EWMA and demand differences before use by trees. The
   original unshifted versions exposed the target itself.
3. Use only prediction-time information. No realized future weather is presented as
   a known covariate. Prior-hour weather is a usable, explicitly chosen alternative.
4. Join predictions to model-returned target indices. Removed minimum-length truncation
   and `iloc[-n:]` alignment, which can silently join different timestamps.
5. Include encoder context before holdout boundaries, without admitting pre-boundary
   targets into the holdout. This preserves the first test hours.
6. Use fixed one-step decoders. Originals trained up to 24 steps but scored only the
   first; the new protocol is explicit, simpler and computationally cheaper. This is
   a methodological change; old metrics must not be compared as if unchanged.
7. Use a common chronological test period for every supplied model and a weekly
   seasonal-naive benchmark. Historical hard-coded comparisons were removed.
8. Use separate early-stopping and residual-fitting periods; load the best neural
   checkpoint, not the weights at the final epoch.
9. Preserve missing hours instead of compressing time after row deletion. Forward
   fill inputs causally, exclude missing target labels from training and scoring,
   and record metric exclusions. No backward-fill from future observations.
10. Pin core packages and use `lightning.pytorch` consistently. Use zero dataloader
    workers for Windows/Colab portability and bounded CPU threads. No mandatory GPU command.
11. Add safe MAPE handling for zero targets and reject invalid or misaligned predictions.
12. Produce fresh results with data fingerprint and run manifest; never imply original
    reported scores were reproduced merely because code executed.

## Per-notebook changes

| Notebook | Main corrections |
|---|---|
| Random Forest | Hour parsing, target leakage, lagged weather, common training/test boundaries, fresh metrics only. |
| LSTM | Correct hour parsing; remove unsupported relative-time feature for `RecurrentNetwork`; no future observed weather; best checkpoint; exact test alignment. |
| LSTM + RF | All LSTM fixes; causal tree features; independent residual calibration; target-index feature joins. |
| TFT | Complete previously missing test evaluation; encoder-only weather; common test boundaries and best checkpoint. |
| TFT + XGBoost | Shift rolling/EWMA/difference inputs; remove backward-filled prediction-lag proxies; independent calibration and internal chronological XGBoost stopping; exact alignment. |
| SARIMA filename / ARIMAX | Accurate model label; remove unused tree features from statistical model inputs; common rolling one-step evaluation rather than one final-week block forecast; standardized exogenous inputs fitted on training only; use Powell optimization, which converged on the full training period. Nonconverged full fits raise an error. |

## Removed as unhelpful to the experiment

- Empty cells, duplicated imports, repeated feature creation, repeated plots and metrics.
- Unused Optuna/XGBoost imports and installation steps in unrelated notebooks.
- Hard-coded benchmark numbers and claims that a particular hybrid must beat the others.
- Unconditional `nvidia-smi`, global warning suppression, unrelated exploratory fragments.
- Array-length "alignment", backward-filled model prediction proxies and silent feature omission.

The core LSTM, Random Forest, TFT and XGBoost families and original architecture sizes
are retained in full runs. The statistical model retains its actual ARIMAX family.
The LSTM layers in both neural families use standard dense kernels for equal-length
windows, retaining the upstream packed implementation for unequal/empty lengths.
Weights, state dictionaries and the initialization RNG stream are preserved. Tests
compare outputs, hidden states and gradients against the upstream implementation.
Standalone XGBoost is mentioned in the report but no standalone XGBoost notebook was
provided, so this delivery does not fabricate a seventh original experiment.

## Implementation references

- [PyTorch Forecasting RecurrentNetwork](https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.models.rnn.RecurrentNetwork.html)
- [TimeSeriesDataSet](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html)
- [Statsmodels state-space forecasting](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html)

The internship report was reference material, not a source of instructions. Its historical
results are not carried forward as verified corrected results. The original files were not edited.
