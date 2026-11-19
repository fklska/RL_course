import unittest

import gymnasium as gym
from minigrid.core.world_object import Key

from scripts.bfs import (
    CLOSED, DIRECTIONS, DROP, FORWARD, LOCKED, OPEN, PICKUP, TOGGLE,
    ENV_ID, evaluate, read_world, replay_result, solve,
    transition_baseline, transition_full,
)


class TransitionTests(unittest.TestCase):
    def state(self, direction=0, carrying=None, items=(), doors=()):
        return ((2, 2), direction, carrying, doors, frozenset(items))

    def test_key_pickup_from_front_or_below(self):
        for facing in range(4):
            for side, (dx, dy) in enumerate(DIRECTIONS):
                with self.subTest(facing=facing, side=side):
                    position = (2 + dx, 2 + dy)
                    state = self.state(facing, items=((position, ("key", "red")),))
                    full, _ = transition_full(state, PICKUP, set(), (), ())
                    baseline, success = transition_baseline(state, PICKUP, set(), (), ())
                    expected = facing == side or side == 1
                    self.assertEqual(full[2], ("key", "red") if facing == side else None)
                    self.assertEqual(baseline[2], ("key", "red") if expected else None)
                    self.assertEqual(bool(baseline[4]), not expected)
                    self.assertEqual(baseline[:2], state[:2])
                    self.assertFalse(success)

    def test_locked_door_keys(self):
        for key in (None, ("key", "red"), ("key", "blue"), ("ball", "red")):
            with self.subTest(key=key):
                state = self.state(carrying=key, doors=(LOCKED,))
                full, _ = transition_full(state, TOGGLE, set(), ((3, 2),), ("red",))
                baseline, _ = transition_baseline(state, TOGGLE, set(), ((3, 2),), ("red",))
                self.assertEqual(full[3], (OPEN,) if key == ("key", "red") else (LOCKED,))
                self.assertEqual(baseline, full)

    def test_occupied_hands_prevent_pickup(self):
        state = self.state(carrying=("key", "blue"), items=(((2, 3), ("key", "red")),))
        for model in (transition_full, transition_baseline):
            self.assertEqual(model(state, PICKUP, set(), (), ()), (state, False))

    def test_ball_still_requires_facing(self):
        for direction in range(4):
            for position in ((3, 2), (2, 3)):
                state = self.state(direction, items=((position, ("ball", "red")),))
                self.assertEqual(
                    transition_baseline(state, PICKUP, set(), (), ()),
                    transition_full(state, PICKUP, set(), (), ()),
                )

    def test_front_key_has_priority_over_below(self):
        items = (((3, 2), ("key", "red")), ((2, 3), ("key", "blue")))
        child, _ = transition_baseline(self.state(items=items), PICKUP, set(), (), ())
        self.assertEqual(child[2], ("key", "red"))
        self.assertIn(((2, 3), ("key", "blue")), child[4])

    def test_other_actions_keep_full_rules(self):
        for door_state in (OPEN, CLOSED, LOCKED):
            state = self.state(carrying=("key", "blue"), doors=(door_state,))
            for action in (0, 1, FORWARD, DROP):
                self.assertEqual(
                    transition_baseline(state, action, set(), ((3, 2),), ("red",)),
                    transition_full(state, action, set(), ((3, 2),), ("red",)),
                )
        for door_state in (OPEN, CLOSED):
            state = self.state(carrying=("key", "blue"), doors=(door_state,))
            self.assertEqual(
                transition_baseline(state, TOGGLE, set(), ((3, 2),), ("red",)),
                transition_full(state, TOGGLE, set(), ((3, 2),), ("red",)),
            )


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.env = gym.make(ENV_ID)
        self.env.reset(seed=0)
        self.addCleanup(self.env.close)

    def test_empty_and_incomplete_plans_are_failure(self):
        for actions in ([], [0, 1]):
            with self.subTest(actions=actions):
                prediction = evaluate(self.env, actions)
                self.assertFalse(prediction.success)
                self.assertFalse(prediction.terminated)
                self.assertFalse(prediction.truncated)
                self.assertEqual(prediction, replay_result(0, actions))

    def test_same_plans_against_real_environment(self):
        initial = read_world(self.env)
        for planner in (transition_full, transition_baseline):
            actions = solve(self.env, planner)
            self.assertIsNotNone(actions)
            real = replay_result(0, actions)
            self.assertEqual(evaluate(self.env, actions), real)
            baseline = evaluate(self.env, actions, transition_baseline)
            if planner is transition_baseline:
                self.assertTrue(baseline.success)
        self.assertEqual(read_world(self.env), initial)
        self.assertEqual(self.env.unwrapped.step_count, 0)

    def test_reads_carried_key_from_environment(self):
        self.env.unwrapped.carrying = Key("red")
        self.assertEqual(read_world(self.env)[3][2], ("key", "red"))

    def test_timeout_and_success_at_step_limit(self):
        plan = solve(self.env)
        for limit, actions in ((1, [0, 0]), (len(plan), plan + [0])):
            with self.subTest(limit=limit):
                self.env.reset(seed=0)
                self.env.unwrapped.max_steps = limit
                prediction = evaluate(self.env, actions)
                steps = 0
                for action in actions:
                    _, reward, terminated, truncated, _ = self.env.step(action)
                    steps += 1
                    if terminated or truncated:
                        break
                self.assertEqual(prediction.steps, steps)
                self.assertEqual(prediction.terminated, terminated)
                self.assertEqual(prediction.truncated, truncated)
                self.assertEqual(prediction.success, bool(terminated and reward > 0))
                self.assertEqual(prediction.final_state, read_world(self.env)[3])

    def test_remaining_budget_from_intermediate_state(self):
        self.env.unwrapped.max_steps = 2
        self.env.step(0)
        prediction = evaluate(self.env, [0, 0])
        self.assertEqual(prediction.steps, 1)
        self.assertTrue(prediction.truncated)
        self.assertEqual(self.env.unwrapped.step_count, 1)

    def test_invalid_actions_are_rejected(self):
        for actions in ([6], [-1], ["pickup"], [True]):
            with self.subTest(actions=actions):
                with self.assertRaises(ValueError):
                    evaluate(self.env, actions)


if __name__ == "__main__":
    unittest.main()
