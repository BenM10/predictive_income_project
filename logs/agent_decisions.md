# Agent Decisions Log

| Date | Decision | Context | Alternative Considered | Justification |\n|---|---|---|---|---|\n| 2026-02-18 | Set `na_values='?'` and `skipinitialspace=True` | Dataset uses '?' for missing values and has leading spaces after commas. | Manual string stripping/replacing after load. | Built-in pandas parameters are more efficient and cleaner. |
| 2026-02-18 | Skip first row of `adult.test` | The test file has an informal header line that causes read errors. | None. | Necessary for valid CSV parsing. |
| 2026-02-18 | Use manually defined column names | Raw files lack headers. | Inferring from code (unreliable). | Accuracy based on dataset documentation. |
| 2026-02-19 | Removed semantic column explanations from EDA | Proposed explaining column meanings before documentation verification | Including unverified semantic explanations | Restricted by user because documentation had not yet been verified |
| 2026-02-19 | Normalised income labels by removing trailing periods | Duplicate income classes found due to formatting differences in test set | Encoding labels as-is | Ensure a valid binary classification target across combined dataset |
| 2026-02-19 | Dropped `fnlwgt` column | Feature represents census sampling weight rather than individual attribute | Scaling the feature | Prevent model distortion from non-predictive sampling weights |
| 2026-02-19 | Use in-depth prompts for new notebooks | Agent struggled editing notebook in place | Serial manual code entries | Minimise manual code entries and improve consistency across notebook creation |
