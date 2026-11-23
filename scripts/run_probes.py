import argparse
import csv
from pathlib import Path

import gymnasium as gym
from minigrid.core.world_object import Key

from scripts.bfs import (
    DIRECTIONS, ENV_ID, PICKUP, TOGGLE, Evaluation, evaluate, read_world,
    transition_baseline, transition_full,
)

MODELS = (("full", transition_full), ("baseline", transition_baseline))


def find_open_cell(world):
    for x in range(1, world.width - 1):
        for y in range(1, world.height - 1):
            cells = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            if all(world.grid.get(cx, cy) is None for cx, cy in cells):
                return x, y
    raise RuntimeError("No empty cell with four empty neighbours")


def find_door(world):
    for x in range(1, world.width - 1):
        for y in range(1, world.height - 1):
            door = world.grid.get(x, y)
            if door is None or door.type != "door":
                continue
            for direction, (dx, dy) in enumerate(DIRECTIONS):
                agent_pos = (x - dx, y - dy)
                if world.grid.get(*agent_pos) is None:
                    return door, agent_pos, direction
    raise RuntimeError("No door with an adjacent empty cell")


def direction_case(seed, direction):
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    world = env.unwrapped
    position = find_open_cell(world)
    world.agent_pos = position
    world.agent_dir = direction
    world.carrying = None
    world.step_count = 0
    world.grid.set(position[0], position[1] + 1, Key("red"))
    return env


def color_case(seed, matching):
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    world = env.unwrapped
    door, agent_pos, direction = find_door(world)
    door.is_open = False
    door.is_locked = True
    world.agent_pos = agent_pos
    world.agent_dir = direction
    color = door.color if matching else ("blue" if door.color != "blue" else "red")
    world.carrying = Key(color)
    world.step_count = 0
    return env


def compare_case(env, action):
    predictions = {name: evaluate(env, [action], model) for name, model in MODELS}
    _, reward, terminated, truncated, _ = env.step(action)
    real = Evaluation(
        bool(terminated and reward > 0), 1, bool(terminated), bool(truncated),
        read_world(env)[3],
    )
    return {name: prediction == real for name, prediction in predictions.items()}


def run_direction_probe(seeds=(3, 4, 5)):
    result = {"n": 0, "full": 0, "baseline": 0}
    for seed in seeds:
        for direction in range(4):
            env = direction_case(seed, direction)
            try:
                matches = compare_case(env, PICKUP)
                result["n"] += 1
                for name, _ in MODELS:
                    result[name] += matches[name]
            finally:
                env.close()
    return result


def run_color_probe(seeds=range(3, 9)):
    result = {"n": 0, "full": 0, "baseline": 0}
    for seed in seeds:
        for matching in (True, False):
            env = color_case(seed, matching)
            try:
                matches = compare_case(env, TOGGLE)
                result["n"] += 1
                for name, _ in MODELS:
                    result[name] += matches[name]
            finally:
                env.close()
    return result


def make_rows(direction, color):
    rows = []
    explanations = {
        "pickup_direction_full": (
            "На каждой карте ключ находится в соседней клетке снизу; проверены четыре направления взгляда. "
            "Полная модель разрешает pickup только при взгляде на ключ."
        ),
        "pickup_direction_baseline": (
            "На каждой карте ключ находится в соседней клетке снизу; проверены четыре направления взгляда. "
            "При взгляде вниз ключ перед агентом и предсказания совпадают; при остальных трёх направлениях "
            "baseline ошибочно подбирает ключ снизу независимо от взгляда, а MiniGrid не подбирает его."
        ),
        "key_color_full": (
            "Для каждой карты проверены подходящий и неподходящий ключ перед запертой дверью. "
            "Полная модель открывает дверь только ключом совпадающего цвета."
        ),
        "key_color_baseline": (
            "Для каждой карты проверены подходящий и неподходящий ключ перед запертой дверью. "
            "Текущая baseline сохраняет проверку цвета ключа; этот зонд является контрольным, "
            "а не проверкой исключённого из baseline правила."
        ),
    }
    scope = (
        "Проверены одношаговые переходы в специально подготовленных состояниях, "
        "а не полные траектории человеческой разметки. Совпадение означает равенство "
        "успеха, числа шагов, флагов завершения и конечного состояния с реальной MiniGrid. "
        "Вывод ограничен проверенными случаями."
    )
    for probe, result, rule in (
        ("pickup_direction", direction, "подбор ключа только с клетки прямо перед агентом"),
        ("key_color", color, "открытие запертой двери только ключом совпадающего цвета"),
    ):
        for name, _ in MODELS:
            matches = result[name]
            n = result["n"]
            rows.append({
                "probe": f"{probe}_{name}",
                "n": n,
                "metric": "agreement_with_env",
                "value": matches / n,
                "interpretation": (
                    f"Совпадений с реальной MiniGrid: {matches} из {n}; расхождений: {n - matches}. "
                    f"Проверялось правило: {rule}. {explanations[f'{probe}_{name}']} {scope}"
                ),
            })
    return rows


def write_results(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["probe", "n", "metric", "value", "interpretation"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("audit/probe_results.csv"))
    args = parser.parse_args()
    direction = run_direction_probe()
    color = run_color_probe()
    rows = make_rows(direction, color)
    write_results(args.output, rows)
    for row in rows:
        print(f"{row['probe']}: value={row['value']} n={row['n']}")
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
