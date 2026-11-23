import unittest

from scripts.run_probes import make_rows, run_color_probe, run_direction_probe


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.direction = run_direction_probe()
        cls.color = run_color_probe()

    def test_direction_probe_against_real_environment(self):
        self.assertEqual(self.direction, {"n": 12, "full": 12, "baseline": 3})

    def test_color_probe_against_real_environment(self):
        self.assertEqual(self.color, {"n": 12, "full": 12, "baseline": 12})

    def test_result_rows_have_required_fields(self):
        rows = make_rows(self.direction, self.color)
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(set(row), {"probe", "n", "metric", "value", "interpretation"})
            self.assertGreater(row["n"], 0)
            self.assertGreaterEqual(row["value"], 0)
            self.assertLessEqual(row["value"], 1)


if __name__ == "__main__":
    unittest.main()
