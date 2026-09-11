# Financial Fraud Detection & Transaction Risk Analytics
### Portfolio Project 004

**Which transactions are genuinely suspicious, how much money is at risk, and which transactions should we investigate first?**

An end-to-end fraud-risk analytics project covering imbalanced classification, point-in-time behavioral features, anomaly detection, explainability, and business-value-oriented investigation prioritization. The data is synthetic and causally structured for portfolio demonstration.

## Pipeline

```text
Transactions → Fraud Signals → Risk Probability → Financial Exposure → Investigation Priority → Business Decision
```

## Development dataset

All published results in this repository were produced on the reproducible `dev` scale:

- **720,970 transactions**
- **6,000 customers**
- **2 years of history**
- approximately **0.63% realized fraud rate**
- approximately **$2.06M total fraud value**

The generator also contains a `full` configuration for 1.5M customers / 3 years, but those full-scale results are **not** claimed here.

## What the project demonstrates

- Point-in-time fraud feature engineering with velocity, device, merchant, amount, geographic, and behavioral signals
- Chronological train/validation/test methodology
- Logistic Regression, Random Forest, and XGBoost under severe class imbalance
- PR-AUC, ROC-AUC, Precision@K, Recall@K, and dollar-weighted capture
- Isolation Forest anomaly detection and overlap analysis
- SHAP-based technical attribution plus business-readable reason codes
- Expected Fraud Loss ranking under fixed evaluation-period investigation budgets
- Leakage-safe operating-threshold selection using a separate policy-tuning and policy-holdout split
- SQL risk analytics and Power BI-ready export tables / DAX measures

## Model results

| Model | PR-AUC | ROC-AUC | Recall@2,162 | Precision@2,162 | $ Capture@2,162 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0361 | **0.6848** | **23.82%** | **7.26%** | **52.84%** |
| Random Forest | 0.0296 | 0.6789 | 18.51% | 5.64% | 29.05% |
| XGBoost | **0.0420** | 0.6613 | 20.18% | 6.15% | 40.28% |

The most useful lesson is that there is no single “best model” independent of the business objective. XGBoost has the strongest PR-AUC, while Logistic Regression captures more fraud dollars at the fixed K used in the model-comparison table.

## Anomaly detection

Isolation-Forest-only alerts show a **2.8% fraud rate** versus **0.61% overall** in the model-test slice — about **4.6× enrichment**. This demonstrates how unsupervised signals can complement a supervised fraud model.

## Flagship business result: prioritize dollars, not only probability

The corrected policy analysis uses the later half of the chronological model-test slice as a final policy holdout. Ranking cases by:

`P(fraud) × transaction value × expected loss rate`

protects more simulated fraud dollars than ranking by fraud probability alone under the same review budget.

| Review budget (policy holdout) | Expected-loss ranking | Probability-only ranking | Lift |
|---:|---:|---:|---:|
| 250 | $125,669 | $57,575 | **+118.3%** |
| 500 | $150,466 | $78,851 | **+90.8%** |
| 1,000 | $176,685 | $109,932 | **+60.7%** |
| 1,500 | $203,453 | $129,044 | **+57.7%** |
| 2,500 | $217,268 | $191,458 | **+13.5%** |

These are retrospective simulation results under an assumed 85% loss rate, not realized production savings.

## Leakage-safe threshold optimization

The earlier version selected an operating threshold on the same slice used for reporting. That has been corrected. The model-test period is split chronologically into an earlier **policy-tuning** half and a later **policy-holdout** half.

The tuning slice selected a threshold of **0.75**. Frozen before touching the later holdout, that threshold produced estimated net value of **$99,873** on the policy holdout versus **$64,262** at a default 0.5 cutoff — about **+55.4%** under the project's explicit investigation/friction/loss assumptions.

See `docs/POLICY_EVALUATION.md` and `src/models/artifacts/policy_evaluation.json`.

## Explainability example

A flagged transaction can be translated from a probability into investigator-facing reason codes such as:

```json
{
  "fraud_probability": 0.9803,
  "risk_score_0_100": 98,
  "reason_codes": [
    "3 failed attempts in the prior 24 hours before this approval",
    "Transaction amount far outside this customer's normal range",
    "Transaction location 10591 km from previous transaction",
    "New, previously unseen device",
    "High-risk merchant category (risk score 2.7)"
  ]
}
```

The reason-code layer includes guardrails so plain-language descriptions remain semantically valid (for example, an old account is never described as “relatively new”).

## Repository layout

```text
data/                   lightweight samples / generated outputs
src/data_generation/    reproducible synthetic data generator
src/features/           leakage-aware point-in-time feature engineering
src/models/             supervised models + result artifacts
src/anomaly/            Isolation Forest analysis
src/explainability/     SHAP + business reason codes
src/optimization/       investigation ranking + policy evaluation
sql/                    fraud/risk analysis queries
dashboards/              Power BI-ready exports and DAX measures
docs/                   data dictionary and methodology
tests/                  publication-integrity tests
```

## Reproduce

Install dependencies from `requirements.txt`, then run the pipeline in order:

```bash
python src/data_generation/generate_entities.py
python src/data_generation/generate_transactions.py
python src/data_generation/generate_outcomes.py
python src/features/build_features.py
python src/models/train_models.py
python src/anomaly/isolation_forest.py
python src/explainability/explain.py
python src/optimization/investigation_optimizer.py
python dashboards/export_powerbi_tables.py
python -m unittest discover -s tests -v
```

## Publication note

This is a synthetic portfolio project. Model metrics, money-protected estimates, and net-business-value estimates are retrospective simulation/evaluation outputs, not claims of live financial performance. See `PUBLICATION_AUDIT.md` for the final audit.
