import copy
from html.parser import HTMLParser
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
from PIL import Image

from scripts.bfs import ENV_ID, solve, transition_baseline
from scripts.export_predictions import (
    PUBLIC_FIELDS, REPOSITORY, export_predictions, validate_destination, validate_slice,
)
from scripts.prepare_audit import (
    ACTION_NAMES, case_html, index_html, make_actions, map_image,
    predicted_states, read_jsonl, write_text_once,
)


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = read_jsonl(REPOSITORY / "audit/slice.jsonl")
        cls.configurations = read_jsonl(REPOSITORY / "data/configurations.jsonl")

    def test_public_records_match_configurations_and_human_text(self):
        validate_slice(self.records)
        self.assertEqual(len(self.records), 52)
        self.assertEqual([(r["id"], r["group"]) for r in self.records],
                         [(r["id"], r["group"]) for r in self.configurations])
        self.assertEqual(sum(r["group"] == "core" for r in self.records), 20)
        forbidden = {"verdict", "predicted_success", "reward", "source", "plan_source", "model", "success", "terminated", "truncated"}
        for record in self.records:
            self.assertEqual(set(record), PUBLIC_FIELDS)
            self.assertFalse(forbidden & set(record))
            self.assertTrue(record["actions"])
            for step, action in enumerate(record["actions"], start=1):
                self.assertIn(f"{step}. {ACTION_NAMES[action]} —", record["content"])

    def test_all_52_plans_are_complete_baseline_plans(self):
        for record in self.records:
            env = gym.make(record["env_id"])
            try:
                env.reset(seed=record["seed"])
                expected = solve(env, transition_baseline)
                self.assertIsNotNone(expected, record["id"])
                self.assertEqual(record["actions"], expected, record["id"])
            finally:
                env.close()

    def test_all_cards_and_maps_exist_and_match_the_slice(self):
        materials = REPOSITORY / "audit/materials"
        self.assertEqual((materials / "index.html").read_text(encoding="utf-8"), index_html(self.records))
        for index, record in enumerate(self.records):
            previous_id = self.records[index - 1]["id"] if index else None
            next_id = self.records[index + 1]["id"] if index + 1 < len(self.records) else None
            page = (REPOSITORY / "audit" / record["materials"]["page"]).read_text(encoding="utf-8")
            self.assertEqual(page, case_html(record, previous_id, next_id))
            HTMLParser().feed(page)
            with Image.open(REPOSITORY / "audit" / record["materials"]["map"]) as image:
                self.assertEqual(image.size, (record["initial_state"]["width"] * 40 + 28,
                                              record["initial_state"]["height"] * 40 + 28))
                env = gym.make(record["env_id"])
                try:
                    env.reset(seed=record["seed"])
                    self.assertEqual(image.convert("RGB").tobytes(), map_image(env, actions=record["actions"]).tobytes())
                finally:
                    env.close()

    def test_tampered_text_actions_and_verdict_leaks_are_rejected(self):
        for change in ("text", "action", "verdict"):
            with self.subTest(change=change):
                records = copy.deepcopy(self.records)
                if change == "text":
                    records[0]["content"] += "changed"
                elif change == "action":
                    records[0]["actions"][0] = (records[0]["actions"][0] + 1) % 6
                else:
                    records[0]["verdict"] = "success"
                with self.assertRaises(ValueError):
                    validate_slice(records)

    def test_private_outputs_cannot_be_written_inside_repository(self):
        for path in (REPOSITORY / "audit/scorer.jsonl", REPOSITORY / "private/scorer.jsonl"):
            with self.assertRaises(ValueError):
                validate_destination(path)
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(validate_destination(Path(folder) / "scorer.jsonl"),
                             (Path(folder) / "scorer.jsonl").resolve())

    def test_export_keeps_id_and_fixed_actions_without_replanning(self):
        # Synthetic verdict only: never expose or assert real held-out labels.
        record = {"id": "test-id", "seed": 0, "actions": [0, 1]}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "scorer.jsonl"
            with patch("scripts.export_predictions.validate_slice"), \
                 patch("scripts.export_predictions.evaluate", return_value=SimpleNamespace(success=False)) as evaluator:
                self.assertEqual(export_predictions([record], output), 1)
                self.assertEqual(evaluator.call_args.args[1], [0, 1])
            self.assertEqual(read_jsonl(output), [{"id": "test-id", "verdict": "failure"}])

    def test_baseline_plan_is_not_shortened_or_replaced(self):
        with patch("scripts.prepare_audit.solve", return_value=[0, 1, 2, 3]) as planner:
            self.assertEqual(make_actions(None), [0, 1, 2, 3])
            planner.assert_called_once_with(None, transition_baseline)
        with patch("scripts.prepare_audit.solve", return_value=None):
            with self.assertRaises(RuntimeError):
                make_actions(None)

    def test_predicted_trace_uses_model_without_replaying_environment(self):
        env = gym.make(ENV_ID)
        try:
            env.reset(seed=0)
            actions = solve(env, transition_baseline)
            states = predicted_states(env, actions)
            self.assertEqual(len(states), len(actions) + 1)
            self.assertEqual(states[-1][2][0], "ball")
            self.assertEqual(env.unwrapped.step_count, 0)
            self.assertIsNone(env.unwrapped.carrying)
        finally:
            env.close()

    def test_changed_human_materials_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "card.html"
            write_text_once(path, "original")
            write_text_once(path, "original")
            with self.assertRaises(FileExistsError):
                write_text_once(path, "changed")
            self.assertEqual(path.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
