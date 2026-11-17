"""Prepare public initial states without running planners or scorers."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import gymnasium as gym

from scripts.bfs import ENV_ID

EXCLUDED_SEEDS = frozenset((0, 1, 2))
DEFAULT_OUTPUT = Path("data/configurations.jsonl")


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def state_fingerprint(initial_state):
    payload = {"env_id": ENV_ID, "initial_state": initial_state}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def configuration_for_seed(seed):
    env = gym.make(ENV_ID)
    try:
        env.reset(seed=seed)
        world = env.unwrapped
        initial_state = {
            "width": int(world.width),
            "height": int(world.height),
            # MiniGrid encoding is indexed [x][y][type, color, state].
            "grid": world.grid.encode().tolist(),
            "agent_pos": [int(x) for x in world.agent_pos],
            "agent_dir": int(world.agent_dir),
            "carrying": None if world.carrying is None else {
                "type": world.carrying.type, "color": world.carrying.color,
            },
            "mission": world.mission,
            "max_steps": int(world.max_steps),
            "step_count": int(world.step_count),
        }
        return {
            "schema_version": 1,
            "env_id": ENV_ID,
            "seed": seed,
            "versions": {name: version(name) for name in ("gymnasium", "minigrid")},
            "fingerprint": state_fingerprint(initial_state),
            "initial_state": initial_state,
        }
    finally:
        env.close()


def generate_records(count=52, core_count=20, start_seed=3, max_candidates=10000):
    if count < 1 or not 0 <= core_count <= count or start_seed < 0 or max_candidates < count:
        raise ValueError("Invalid count, core_count, start_seed or max_candidates")
    records, seen = [], set()
    for seed in range(start_seed, start_seed + max_candidates):
        if seed in EXCLUDED_SEEDS:
            continue
        record = configuration_for_seed(seed)
        if record["fingerprint"] in seen:
            continue
        seen.add(record["fingerprint"])
        index = len(records) + 1
        record.update(id=f"K04-{index:03d}", group="core" if index <= core_count else "second")
        records.append(record)
        if len(records) == count:
            return records
    raise RuntimeError("Not enough unique configurations within max_candidates")


def verify_records(records, count=52, core_count=20):
    if len(records) != count:
        raise ValueError(f"Expected {count} records, got {len(records)}")
    fingerprints, seeds = set(), set()
    for index, record in enumerate(records, start=1):
        seed = record["seed"]
        if type(seed) is not int or seed < 0 or seed in EXCLUDED_SEEDS or seed in seeds:
            raise ValueError(f"Invalid, excluded or duplicate seed at row {index}")
        seeds.add(seed)
        expected = configuration_for_seed(seed)
        expected.update(id=f"K04-{index:03d}", group="core" if index <= core_count else "second")
        if record != expected:
            raise ValueError(f"Stored configuration differs from seed replay at row {index}")
        if record["fingerprint"] in fingerprints:
            raise ValueError(f"Duplicate initial state at row {index}")
        fingerprints.add(record["fingerprint"])
    return len(records)


def write_records(path, records):
    content = "".join(canonical_json(record) + "\n" for record in records)
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return False
        raise FileExistsError(f"Refusing to replace a different dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(content)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-seed", type=int, default=3)
    parser.add_argument("--verify", action="store_true", help="Verify a stored dataset without changing it")
    args = parser.parse_args()
    if args.verify:
        with args.output.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source]
        verify_records(records)
        print("Verified 52 unique configurations: 20 core, 32 second; exact seed replay")
    else:
        records = generate_records(start_seed=args.start_seed)
        verify_records(records)
        created = write_records(args.output, records)
        print(f"{'Created' if created else 'Unchanged'} {args.output}: 52 unique configurations, 20 core, 32 second")
        print(f"IDs K04-001..K04-052; seeds {min(r['seed'] for r in records)}..{max(r['seed'] for r in records)}")


if __name__ == "__main__":
    main()
