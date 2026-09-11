# Leakage-Safe Policy Evaluation

The fraud classifier uses a chronological 70/15/15 train/validation/model-test split. For business-policy development, the original model-test slice is then split chronologically into two equal halves:

- Policy tuning: **54,073 rows**, 2025-09-12 06:40:35 to 2025-11-06 08:19:07
- Policy holdout: **54,073 rows**, 2025-11-06 08:19:40 to 2025-12-30 23:58:51

The operating threshold is selected only on the policy-tuning half, then frozen before evaluation on the later holdout.

**Selected threshold on tuning slice:** 0.75

**Later holdout performance at frozen threshold:**
- Net business value: **$99,873**
- Default 0.5 threshold: **$64,262**
- Improvement: **55.4%**

At a 500-case review budget on the later holdout, expected-loss ranking protects **$150,466** versus **$78,851** for probability-only ranking, a **90.8%** lift under the same review capacity.

These are retrospective simulated outcomes, not realized production savings.
