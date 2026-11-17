import argparse
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import minigrid

ENV_ID = "BabyAI-UnlockToUnlock-v0"
LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE = range(6)
DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
OPEN, CLOSED, LOCKED = range(3)


def read_world(env):
    world = env.unwrapped
    walls = set()
    doors = {}
    objects = {}
    for x in range(world.width):
        for y in range(world.height):
            cell = world.grid.get(x, y)
            if cell is None:
                continue

            if cell.type == "wall":
                walls.add((x, y))
            elif cell.type == "door":
                state = LOCKED if cell.is_locked else OPEN if cell.is_open else CLOSED
                doors[(x, y)] = (cell.color, state)
            else:
                objects[(x, y)] = (cell.type, cell.color)

    door_positions = tuple(sorted(doors))
    door_colors = tuple(doors[p][0] for p in door_positions)
    start = (
        tuple(world.agent_pos),
        world.agent_dir,
        (world.carrying.type, world.carrying.color) if world.carrying is not None else None,
        tuple(doors[p][1] for p in door_positions),
        frozenset(objects.items()),
    )
    return walls, door_positions, door_colors, start


def transition_full(state, action, walls, door_positions, door_colors):
    position, direction, carrying, door_states, objects = state
    dx, dy = DIRECTIONS[direction]
    front = (position[0] + dx, position[1] + dy)
    items = dict(objects)
    door = door_positions.index(front) if front in door_positions else None
    if action == LEFT:
        return (position, (direction - 1) % 4, carrying, door_states, objects), False

    if action == RIGHT:
        return (position, (direction + 1) % 4, carrying, door_states, objects), False

    if action == FORWARD:
        blocked = front in walls or front in items or (door is not None and door_states[door] != OPEN)
        if blocked:
            return state, False

        return (front, direction, carrying, door_states, objects), False

    if action == PICKUP:
        if carrying is not None or front not in items:
            return state, False

        item = items.pop(front)
        return (position, direction, item, door_states, frozenset(items.items())), item[0] == "ball"

    if action == DROP:
        if carrying is None or front in walls or front in items or door is not None:
            return state, False

        items[front] = carrying
        return (position, direction, None, door_states, frozenset(items.items())), False

    if action == TOGGLE and door is not None:
        states = list(door_states)
        if states[door] == LOCKED:
            if carrying != ("key", door_colors[door]):
                return state, False

            states[door] = OPEN
        else:
            states[door] = CLOSED if states[door] == OPEN else OPEN

        return (position, direction, carrying, tuple(states), objects), False

    return state, False


def transition_baseline(state, action, walls, door_positions, door_colors):
    """Ignore key colors and allow adjacent keys to be picked up without turning."""
    position, direction, carrying, door_states, objects = state
    if action == PICKUP and carrying is None:
        items = dict(objects)
        # Prefer the front cell (including normal ball pickup), then right,
        # behind, and left. Only keys can be picked up without facing them.
        for offset in range(4):
            dx, dy = DIRECTIONS[(direction + offset) % 4]
            target = (position[0] + dx, position[1] + dy)
            item = items.get(target)
            if item is not None and (offset == 0 or item[0] == "key"):
                items.pop(target)
                child = (position, direction, item, door_states, frozenset(items.items()))
                return child, item[0] == "ball"

    if action == TOGGLE and carrying is not None and carrying[0] == "key":
        # Reuse full door mechanics, treating every door as matching this key.
        door_colors = (carrying[1],) * len(door_colors)
    return transition_full(state, action, walls, door_positions, door_colors)


def solve(env, transition_fn=transition_full):
    walls, door_positions, door_colors, start = read_world(env)
    parents = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        for action in (LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE):
            child, success = transition_fn(state, action, walls, door_positions, door_colors)
            if success:
                actions = [action]
                while parents[state] is not None:
                    state, previous = parents[state]
                    actions.append(previous)

                return actions[::-1]

            if child not in parents:
                parents[child] = (state, action)
                queue.append(child)

    return None


@dataclass(frozen=True)
class Evaluation:
    success: bool
    steps: int
    terminated: bool
    truncated: bool
    final_state: tuple


def validate_actions(actions):
    actions = tuple(actions)
    if any(not isinstance(action, int) or isinstance(action, bool)
           or action not in range(6) for action in actions):
        raise ValueError("Actions must be integers 0..5 (left through toggle)")
    return actions


def evaluate(env, actions, transition_fn=transition_full):
    """Predict a fixed plan from a nonterminal state without changing env.

    Supports UnlockToUnlock's key/ball/door mechanics, not arbitrary MiniGrid
    missions. Exhausting a plan without picking up the ball means failure,
    even when the environment would allow more actions.
    """
    actions = validate_actions(actions)
    walls, door_positions, door_colors, state = read_world(env)
    world = env.unwrapped
    remaining = max(0, world.max_steps - world.step_count)
    success, terminated, truncated, steps = False, False, remaining == 0, 0
    for action in actions:
        if terminated or truncated:
            break
        state, success = transition_fn(state, action, walls, door_positions, door_colors)
        steps += 1
        terminated = success
        truncated = steps >= remaining
    return Evaluation(success, steps, terminated, truncated, state)


def replay_result(seed, actions):
    """Execute the fixed plan in a fresh real environment with the same seed."""
    actions = validate_actions(actions)
    env = gym.make(ENV_ID)
    try:
        env.reset(seed=seed)
        reward, terminated, truncated, steps = 0.0, False, False, 0
        for action in actions:
            _, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        final_state = read_world(env)[3]
        return Evaluation(bool(terminated and reward > 0), steps,
                          bool(terminated), bool(truncated), final_state)
    finally:
        env.close()


def replay(seed, actions):
    result = replay_result(seed, actions)
    return result.success, result.steps


def audit_plans(seeds, plan_source):
    models = (("full", transition_full), ("baseline", transition_baseline))
    planner = dict(models)[plan_source]
    matches = {name: 0 for name, _ in models}
    evaluated = 0
    for seed in range(seeds):
        env = gym.make(ENV_ID)
        try:
            env.reset(seed=seed)
            actions = solve(env, planner)
            if actions is None:
                print(f"seed={seed} plan_source={plan_source} no path; skipped")
                continue
            real = replay_result(seed, actions)
            evaluated += 1
            for name, model in models:
                prediction = evaluate(env, actions, model)
                match = prediction == real
                matches[name] += match
                print(
                    f"seed={seed} plan_source={plan_source} evaluator={name} "
                    f"predicted_success={prediction.success} real_success={real.success} "
                    f"predicted_steps={prediction.steps} real_steps={real.steps} "
                    f"state_match={prediction.final_state == real.final_state} match={match}"
                )
        finally:
            env.close()
    for name, _ in models:
        print(f"evaluator={name} match {matches[name]}/{evaluated} requested={seeds}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--audit", action="store_true",
                        help="Evaluate the same fixed plan with both models")
    parser.add_argument("--plan-source", choices=("full", "baseline"), default="full")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    if args.audit:
        audit_plans(args.seeds, args.plan_source)
        return
    models = (("full", transition_full), ("baseline", transition_baseline))
    matches = {name: 0 for name, _ in models}
    for seed in range(args.seeds):
        env = gym.make(ENV_ID)
        try:
            env.reset(seed=seed)
            # Both searches read the same environment; neither changes it.
            for name, transition_fn in models:
                actions = solve(env, transition_fn)
                if actions is None:
                    print(f"seed={seed} model={name} no path")
                    continue

                # Each plan is replayed in its own fresh real environment.
                success, steps = replay(seed, actions)
                match = success and steps == len(actions)
                matches[name] += match
                print(
                    f"seed={seed} model={name} predicted={len(actions)} "
                    f"success={success} steps={steps} match={match}"
                )
        finally:
            env.close()

    for name, _ in models:
        print(f"model={name} match {matches[name]}/{args.seeds}")


if __name__ == "__main__":
    main()
