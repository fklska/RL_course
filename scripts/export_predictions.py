"""Export predictions to a curator-held location without printing verdicts."""

import argparse
from pathlib import Path

import gymnasium as gym

from scripts.bfs import ENV_ID, evaluate, transition_baseline, transition_full, validate_actions
from scripts.generate_configurations import verify_records, write_records
from scripts.prepare_audit import case_content, read_jsonl

REPOSITORY = Path(__file__).resolve().parents[1]
PUBLIC_FIELDS = {"id", "group", "env_id", "seed", "configuration_fingerprint", "versions", "initial_state", "actions", "content", "materials"}


def validate_slice(records):
    configurations = []
    for record in records:
        if set(record) != PUBLIC_FIELDS:
            raise ValueError("Unexpected or missing public fields in slice")
        validate_actions(record["actions"])
        configurations.append({
            "schema_version": 1, "id": record["id"], "group": record["group"],
            "env_id": record["env_id"], "seed": record["seed"],
            "fingerprint": record["configuration_fingerprint"],
            "versions": record["versions"], "initial_state": record["initial_state"],
        })
    verify_records(configurations)
    for record in records:
        env = gym.make(ENV_ID)
        try:
            env.reset(seed=record["seed"])
            if record["content"] != case_content(env, record["actions"]):
                raise ValueError("Human text differs from the actions and initial state")
        finally:
            env.close()
        expected_materials = {"page": f"materials/{record['id']}.html", "map": f"materials/maps/{record['id']}.png"}
        if record["materials"] != expected_materials:
            raise ValueError("Unexpected material paths")


def validate_destination(path):
    resolved = path.resolve()
    if resolved.is_relative_to(REPOSITORY):
        raise ValueError("Before annotation ends, predictions must be outside the shared repository")
    return resolved


def export_predictions(records, output, model_name="full"):
    output = validate_destination(output)
    if model_name not in ("full", "baseline"):
        raise ValueError("Unknown evaluator")
    validate_slice(records)
    model = transition_full if model_name == "full" else transition_baseline
    predictions = []
    for record in records:
        env = gym.make(ENV_ID)
        try:
            env.reset(seed=record["seed"])
            result = evaluate(env, record["actions"], model)
            predictions.append({"id": record["id"], "verdict": "success" if result.success else "failure"})
        finally:
            env.close()
    write_records(output, predictions)
    return len(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", type=Path, default=Path("audit/slice.jsonl"))
    parser.add_argument("--output", type=Path, required=True, help="Location outside the shared repository, held by the curator")
    parser.add_argument("--model", choices=("full", "baseline"), default="full")
    args = parser.parse_args()
    try:
        count = export_predictions(read_jsonl(args.slice), args.output, args.model)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Exported {count} predictions to {args.output}; verdicts are not displayed")


if __name__ == "__main__":
    main()
