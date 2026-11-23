import unittest

import numpy as np

from scripts.team_metrics import agreement, load_core, vs_humans


class TeamMetricsTests(unittest.TestCase):
    def test_agreement_on_known_labels(self):
        labels = np.array([
            ["success", "success", "success"],
            ["failure", "failure", "failure"],
            ["success", "success", "failure"],
            ["success", "failure", "failure"],
        ])
        result = agreement(labels, n_boot=20)
        self.assertAlmostEqual(result["pairwise_agreement"], 2 / 3)
        self.assertAlmostEqual(result["alpha"], 7 / 18)
        self.assertEqual(result["unanimous_share"], 0.5)
        self.assertEqual(len(result["agreement_ci"]), 2)
        self.assertEqual(len(result["alpha_ci"]), 2)

    def test_vs_humans_uses_majority(self):
        humans = [
            ["success", "success", "success"],
            ["failure", "failure", "failure"],
            ["success", "success", "failure"],
            ["success", "failure", "failure"],
        ]
        verdicts = ["success", "success", "failure", "success"]
        result = vs_humans(verdicts, humans)
        self.assertEqual(result["matches"], 1)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["agreement"], 0.25)
        self.assertEqual(len(result["ci"]), 2)

    def test_actual_core_data_is_complete(self):
        core_ids, annotators, labels, full, baseline = load_core("audit")
        self.assertEqual(len(core_ids), 20)
        self.assertEqual(annotators, ["A1", "A2", "A3"])
        self.assertEqual(np.asarray(labels).shape, (20, 3))
        for name, predictions in (("full", full), ("baseline", baseline)):
            with self.subTest(name=name):
                result = vs_humans([predictions[object_id] for object_id in core_ids], labels)
                self.assertEqual(result["n"], 20)


if __name__ == "__main__":
    unittest.main()
