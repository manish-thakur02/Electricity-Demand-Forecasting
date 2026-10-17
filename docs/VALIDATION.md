# Publication validation — 17 October 2026

The six final prediction files were read locally from the preceding Drive audit extraction. All have 6,552 finite predictions on identical, unique, hourly timestamps and exactly identical observed targets (2024-04-03 00:00 through 2024-12-31 23:00). MAE, MSE, RMSE, R² and MAPE match each saved metrics row to numerical precision. Live Drive metric reads on 17 October agree with the extracted metrics. Source prediction hashes are recorded in SOURCE_PROVENANCE.json; raw observation tables are excluded.

The earlier local private CSV has SHA-256 6732f5cc2352e1bffa729ee788ffa372c5363d10933a48a3adaa93e379d82984, different from the final manifests. Exact final-input identity is unverified; it must be supplied independently. Shared modules are copied byte-for-byte from the recovered local project. Compared with the corrected ZIP, models.py contains an existing CUDA compatibility fix: TFT uses deterministic="warn" on CUDA because its interpolation backward lacks a deterministic implementation. The other six modules match archive text after newline normalization. Final-run source revision linkage remains unverified. Notebook adaptations only add repository-root discovery and correct local-data/layout documentation. Their original hashes are preserved. No saved outputs were added.

All six original forecast images and both rebuilt charts were visually inspected before packaging. The comparison charts use only the six verified model keys. Source figures depict successive one-hour forecasts and preserve source units. No unsupported model series is added.

The original validation record is retained as VALIDATION_2026-10-10.md with available historical logs; its smoke/one-epoch results do not substantiate final training convergence. Exact source linkage to final Colab jobs, checkpoint selection, recovery inference script and complete training logs remain unverified. Current publication check results are recorded in PUBLICATION_CHECKS.txt.

The 21 recovered regression tests passed in the existing project validation environment. Notebook structure and Python syntax checks passed. This is not a clean-install verification.

The 21 recovered regression tests passed in the existing project validation environment. Notebook structure and Python syntax checks passed. This is not a clean-install verification.
