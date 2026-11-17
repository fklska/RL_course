import argparse
from collections import deque

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
        None,
        tuple(doors[p][1] for p in door_positions),
        frozenset(objects.items()),
    )
    return walls, door_positions, door_colors, start


def transition(state, action, walls, door_positions, door_colors):
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


def solve(env):
    walls, door_positions, door_colors, start = read_world(env)
    parents = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        for action in (LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE):
            child, success = transition(state, action, walls, door_positions, door_colors)
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


def replay(seed, actions):
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    reward, terminated, steps = 0.0, False, 0
    for action in actions:
        _, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break

    env.close()
    return terminated and reward > 0, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    matches = 0
    for seed in range(args.seeds):
        env = gym.make(ENV_ID)
        env.reset(seed=seed)
        actions = solve(env)
        env.close()
        if actions is None:
            print(f"seed={seed} no path")
            continue

        success, steps = replay(seed, actions)
        match = success and steps == len(actions)
        matches += match
        print(f"seed={seed} predicted={len(actions)} success={success} steps={steps} match={match}")

    print(f"match {matches}/{args.seeds}")


if __name__ == "__main__":
    main()
