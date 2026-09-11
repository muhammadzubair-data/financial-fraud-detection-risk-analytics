# Project 004 Publication Audit

## Status: GitHub-ready after corrections

This audit addresses the publication issues identified in the original completed package.

### Corrections completed

1. **Investigation budgets are no longer described as daily.** The original ranking code selected the top N cases over the complete evaluation slice, so the project now calls these fixed **evaluation-period review budgets**.
2. **Operating-threshold selection is leakage-safe.** The original 0.65 threshold was selected on the same test slice used to report it. The corrected policy analysis splits the chronological model-test slice into an earlier policy-tuning half and a later policy-holdout half. The threshold is selected on the earlier half, frozen, and then evaluated once on the later half.
3. **Explainability reason-code guard added.** `account_age_days` can no longer produce “Relatively new account” when account age exceeds 90 days. Existing saved sample explanations were sanitized.
4. **Publication-integrity tests added.** Tests verify headline dataset/model results, anomaly enrichment, corrected policy evaluation, reason-code safety, and wording.

### Verified results safe to publish

- Development scale: **720,970 transactions**, **6,000 customers**, **2 years**.
- Fraud prevalence: approximately **0.63%**, with about **$2.06M** total fraud value in the full development dataset.
- Best PR-AUC: **XGBoost 0.0420**.
- Logistic Regression has the strongest ROC-AUC (**0.6848**) and dollar capture at K=2,162 (**52.84%**), illustrating that “best model” depends on the operating objective.
- Isolation-Forest-only alerts have a **2.8% fraud rate** versus **0.61%** overall in the model-test slice (~**4.6x enrichment**).
- On the final policy holdout, expected-loss ranking protects **$150,466** at a 500-case review budget versus **$78,851** for probability-only ranking (**+90.8%**).
- A threshold of **0.75** was selected on the policy-tuning slice. When frozen and applied to the later policy holdout, it produced estimated net business value of **$99,873**, versus **$64,262** at the default 0.5 threshold (**+55.4%**).

### Important interpretation

All monetary “protected” and “net value” figures are retrospective simulation/evaluation results under explicit cost and loss-rate assumptions. They are not claims of realized production savings. The project uses synthetic data and is intended as a portfolio demonstration of methodology.
