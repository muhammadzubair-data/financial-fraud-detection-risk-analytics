import csv
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class PublicationIntegrityTests(unittest.TestCase):
    def test_model_headlines(self):
        rows = json.loads((ROOT/'src/models/artifacts/model_comparison.json').read_text())
        d = {r['model']: r for r in rows}
        self.assertAlmostEqual(d['XGBoost']['pr_auc'], 0.0420, places=4)
        self.assertAlmostEqual(d['LogisticRegression']['roc_auc'], 0.6848, places=4)
        self.assertAlmostEqual(d['LogisticRegression']['dollar_capture_at_2162'], 0.5284, places=4)

    def test_anomaly_enrichment(self):
        a = json.loads((ROOT/'src/models/artifacts/anomaly_summary.json').read_text())
        self.assertGreater(a['fraud_rate_in_iso_only'], a['overall_fraud_rate'])
        self.assertGreater(a['fraud_rate_in_iso_only']/a['overall_fraud_rate'], 4.0)

    def test_policy_holdout_is_temporally_later_and_frozen(self):
        p = json.loads((ROOT/'src/models/artifacts/policy_evaluation.json').read_text())
        self.assertLess(p['methodology']['policy_tuning_end'], p['methodology']['policy_holdout_start'])
        self.assertEqual(p['selected_threshold_on_policy_tuning']['threshold'],
                         p['frozen_threshold_performance_on_policy_holdout']['threshold'])
        self.assertGreater(p['frozen_threshold_performance_on_policy_holdout']['net_business_value'],
                           p['default_0_5_performance_on_policy_holdout']['net_business_value'])

    def test_expected_loss_ranking_adds_value_at_tight_budget(self):
        p = json.loads((ROOT/'src/models/artifacts/policy_evaluation.json').read_text())
        r = {x['investigation_budget']: x for x in p['ranking_strategy_comparison_on_policy_holdout']}
        self.assertGreater(r[500]['money_protected_expected_loss_ranking'],
                           r[500]['money_protected_probability_only_ranking'])
        self.assertGreater(r[500]['dollar_lift_pct_from_expected_loss_ranking'], 50)

    def test_no_invalid_new_account_reason_codes(self):
        exps = json.loads((ROOT/'src/models/artifacts/sample_explanations.json').read_text())
        pat = re.compile(r'Relatively new account \((\d+) days old\)')
        for e in exps:
            for reason in e.get('reason_codes', []):
                m = pat.fullmatch(reason)
                if m:
                    self.assertLessEqual(int(m.group(1)), 90)

    def test_readme_does_not_call_period_budget_daily(self):
        text = (ROOT/'README.md').read_text().lower()
        self.assertNotIn('| daily budget |', text)
        self.assertIn('policy holdout', text)
        self.assertIn('synthetic portfolio project', text)

if __name__ == '__main__':
    unittest.main()
