import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.generate_configurations import (
    configuration_for_seed, generate_records, verify_records, write_records,
)


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = generate_records()

    def test_complete_unique_and_reproducible_dataset(self):
        self.assertEqual(verify_records(self.records), 52)
        self.assertEqual(sum(r["group"] == "core" for r in self.records), 20)
        self.assertEqual(sum(r["group"] == "second" for r in self.records), 32)
        self.assertEqual(len({r["fingerprint"] for r in self.records}), 52)
        self.assertTrue(all(r["seed"] not in (0, 1, 2) for r in self.records))
        self.assertEqual(self.records, generate_records())
        for record in self.records:
            self.assertNotIn("verdict", record)
            self.assertNotIn("success", record)
            self.assertNotIn("actions", record)

    def test_tampering_and_duplicate_ids_are_detected(self):
        for field, value in (("id", "K04-002"), ("group", "second"),
                             ("fingerprint", "wrong"), ("seed", 0)):
            with self.subTest(field=field):
                records = copy.deepcopy(self.records)
                records[0][field] = value
                with self.assertRaises(ValueError):
                    verify_records(records)
        records = copy.deepcopy(self.records)
        records[0]["initial_state"]["agent_dir"] = 99
        with self.assertRaises(ValueError):
            verify_records(records)

    def test_duplicate_states_are_skipped(self):
        first = configuration_for_seed(3)
        second = configuration_for_seed(4)
        duplicate = copy.deepcopy(first)
        duplicate["seed"] = 4
        second["seed"] = 5
        with patch("scripts.generate_configurations.configuration_for_seed",
                   side_effect=[first, duplicate, second]):
            records = generate_records(count=2, core_count=1, max_candidates=3)
        self.assertEqual([r["seed"] for r in records], [3, 5])

    def test_identical_output_is_preserved_and_changed_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "configurations.jsonl"
            self.assertTrue(write_records(path, self.records))
            original = path.read_bytes()
            self.assertFalse(write_records(path, self.records))
            with self.assertRaises(FileExistsError):
                write_records(path, self.records[:-1])
            self.assertEqual(path.read_bytes(), original)

    def test_candidate_limit_does_not_return_partial_dataset(self):
        first = configuration_for_seed(3)
        with patch("scripts.generate_configurations.configuration_for_seed",
                   return_value=first):
            with self.assertRaises(RuntimeError):
                generate_records(count=2, core_count=1, max_candidates=2)


if __name__ == "__main__":
    unittest.main()
