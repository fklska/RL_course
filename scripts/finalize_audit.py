import argparse
import csv
from collections import defaultdict
from pathlib import Path

import gymnasium as gym

from scripts.bfs import ENV_ID, evaluate, transition_baseline, transition_full
from scripts.export_predictions import validate_slice
from scripts.generate_configurations import write_records
from scripts.prepare_audit import read_jsonl


def read_annotations(path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError("annotations.csv is empty")
    delimiter = ";" if lines[0].count(";") > lines[0].count(",") else ","
    return list(csv.DictReader(lines, delimiter=delimiter))


def validate_annotations(records, rows):
    groups = {record["id"]: record["group"] for record in records}
    core_ids = {object_id for object_id, group in groups.items() if group == "core"}
    by_annotator = defaultdict(lambda: {"core": set(), "second": set()})
    seen = set()
    for row in rows:
        object_id = row.get("id", "")
        annotator = row.get("annotator", "")
        if object_id not in groups or not annotator:
            raise ValueError("Unknown object or empty annotator in annotations.csv")
        if row.get("label") not in ("success", "failure"):
            raise ValueError(f"Missing or invalid label for {object_id}, {annotator}")
        if not row.get("elapsed_seconds"):
            raise ValueError(f"Missing elapsed time for {object_id}, {annotator}")
        pair = (object_id, annotator)
        if pair in seen:
            raise ValueError(f"Duplicate annotation for {object_id}, {annotator}")
        seen.add(pair)
        by_annotator[annotator][groups[object_id]].add(object_id)
    if not 2 <= len(by_annotator) <= 4:
        raise ValueError("Expected 2 to 4 annotators")
    for annotator, annotated in by_annotator.items():
        if annotated["core"] != core_ids:
            raise ValueError(f"Incomplete core annotation for {annotator}")
        if not 4 <= len(annotated["second"]) <= 8:
            raise ValueError(f"Expected 4 to 8 second objects for {annotator}")
    return len(rows)


def predict(records, model):
    predictions = []
    for record in records:
        env = gym.make(ENV_ID)
        try:
            env.reset(seed=record["seed"])
            result = evaluate(env, record["actions"], model)
            predictions.append({"id": record["id"], "verdict": "success" if result.success else "failure"})
        finally:
            env.close()
    return predictions


def write_baseline(path, records, predictions):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["id", "verdict"]:
            raise ValueError("Unexpected baseline_predictions.csv headers")
        rows = list(reader)
    expected_ids = [record["id"] for record in records]
    if [row["id"] for row in rows] != expected_ids:
        raise ValueError("baseline_predictions.csv IDs differ from slice.jsonl")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["id", "verdict"])
        writer.writeheader()
        writer.writerows(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=Path("audit"))
    args = parser.parse_args()
    records = read_jsonl(args.audit / "slice.jsonl")
    annotations = read_annotations(args.audit / "annotations.csv")
    annotation_count = validate_annotations(records, annotations)
    validate_slice(records)
    full = predict(records, transition_full)
    baseline = predict(records, transition_baseline)
    write_records(args.audit / "scorer.jsonl", full)
    write_baseline(args.audit / "baseline_predictions.csv", records, baseline)
    print(f"Validated annotations: {annotation_count}")
    print(f"Saved full predictions: {len(full)} to {args.audit / 'scorer.jsonl'}")
    print(f"Saved baseline predictions: {len(baseline)} to {args.audit / 'baseline_predictions.csv'}")


if __name__ == "__main__":
    main()
