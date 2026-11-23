import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def agreement(labels, n_boot=400, seed=7):
    labels = np.asarray(labels)
    n, m = labels.shape
    pairs = np.triu_indices(m, k=1)

    def metrics(sample):
        pairwise = float(np.mean(sample[:, pairs[0]] == sample[:, pairs[1]]))
        _, counts = np.unique(sample, return_counts=True)
        total = sample.size
        de = 1.0 - np.sum(counts * (counts - 1)) / (total * (total - 1))
        alpha = float(1.0 - (1.0 - pairwise) / de) if de > 0 else float("nan")
        return pairwise, alpha

    pairwise, alpha = metrics(labels)
    rng = np.random.default_rng(seed)
    samples = np.array([metrics(labels[rng.integers(0, n, size=n)]) for _ in range(n_boot)])

    def interval(values):
        values = values[np.isfinite(values)]
        if not len(values):
            return float("nan"), float("nan")
        return tuple(float(x) for x in np.percentile(values, [2.5, 97.5]))

    return {
        "pairwise_agreement": pairwise,
        "alpha": alpha,
        "agreement_ci": interval(samples[:, 0]),
        "alpha_ci": interval(samples[:, 1]),
        "unanimous_share": float(np.mean(np.all(labels == labels[:, :1], axis=1))),
    }


def wilson_ci(matches, n):
    if n == 0:
        return float("nan"), float("nan")
    z = 1.96
    rate = matches / n
    denominator = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denominator
    return max(0, center - half), min(1, center + half)


def vs_humans(verdicts, humans):
    matches = 0
    n = 0
    for verdict, votes in zip(verdicts, humans):
        counts = Counter(votes)
        maximum = max(counts.values())
        leaders = [label for label, count in counts.items() if count == maximum]
        if len(leaders) != 1:
            continue
        n += 1
        matches += int(verdict == leaders[0])
    return {
        "agreement": matches / n if n else float("nan"),
        "matches": matches,
        "n": n,
        "ci": wilson_ci(matches, n),
    }


def read_annotations(path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    delimiter = ";" if lines[0].count(";") > lines[0].count(",") else ","
    return list(csv.DictReader(lines, delimiter=delimiter))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_core(audit):
    audit = Path(audit)
    records = read_jsonl(audit / "slice.jsonl")
    core_ids = [record["id"] for record in records if record["group"] == "core"]
    annotations = [row for row in read_annotations(audit / "annotations.csv") if row["id"] in core_ids]
    annotators = sorted({row["annotator"] for row in annotations})
    by_pair = {(row["id"], row["annotator"]): row["label"] for row in annotations}
    labels = []
    for object_id in core_ids:
        row = []
        for annotator in annotators:
            label = by_pair.get((object_id, annotator))
            if label not in ("success", "failure"):
                raise ValueError(f"Missing or invalid label for {object_id}, {annotator}")
            row.append(label)
        labels.append(row)
    full = {row["id"]: row["verdict"] for row in read_jsonl(audit / "scorer.jsonl")}
    with (audit / "baseline_predictions.csv").open(encoding="utf-8-sig", newline="") as source:
        baseline = {row["id"]: row["verdict"] for row in csv.DictReader(source)}
    for object_id in core_ids:
        if full.get(object_id) not in ("success", "failure"):
            raise ValueError(f"Missing full verdict for {object_id}")
        if baseline.get(object_id) not in ("success", "failure"):
            raise ValueError(f"Missing baseline verdict for {object_id}")
    return core_ids, annotators, labels, full, baseline


def format_number(value):
    return "—" if not math.isfinite(value) else f"{value:.4f}"


def format_ci(interval):
    return f"[{format_number(interval[0])}; {format_number(interval[1])}]"


def make_markdown(core_ids, annotators, human_metrics, model_metrics):
    lines = [
        "# Командные метрики K04 — core",
        "",
        f"Расчёт выполнен по {len(core_ids)} объектам: от {core_ids[0]} до {core_ids[-1]}. "
        f"Разметчики: {', '.join(annotators)}.",
        "",
        "## Согласие между людьми",
        "",
        "| Метрика | Значение | 95% bootstrap-интервал |",
        "|---|---:|---:|",
        f"| `pairwise_agreement` | {format_number(human_metrics['pairwise_agreement'])} | {format_ci(human_metrics['agreement_ci'])} |",
        f"| `alpha` | {format_number(human_metrics['alpha'])} | {format_ci(human_metrics['alpha_ci'])} |",
        f"| `unanimous_share` | {format_number(human_metrics['unanimous_share'])} | — |",
        "",
        "## Согласие моделей с человеческим большинством",
        "",
        "| Модель | Совпадения | `n` | `agreement` | 95% интервал Уилсона |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in model_metrics.items():
        lines.append(
            f"| `{name}` | {result['matches']} | {result['n']} | "
            f"{format_number(result['agreement'])} | {format_ci(result['ci'])} |"
        )
    lines.extend([
        "",
        "## Параметры расчёта",
        "",
        "- Человеческая метка определяется большинством голосов; объекты с ничьей исключаются.",
        "- Для `pairwise_agreement` усредняются совпадения всех пар разметчиков.",
        "- `alpha` — номинальная α Криппендорфа.",
        "- Bootstrap: 400 выборок, seed 7, объекты перевыбираются целиком с возвращением; интервалы — перцентили 2.5% и 97.5%.",
        "- Интервал Уилсона рассчитан с z = 1.96.",
        "",
        "Воспроизведение: `python -m scripts.team_metrics`.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=Path("audit"))
    args = parser.parse_args()
    core_ids, annotators, labels, full, baseline = load_core(args.audit)
    human_metrics = agreement(labels)
    model_metrics = {
        "full": vs_humans([full[object_id] for object_id in core_ids], labels),
        "baseline": vs_humans([baseline[object_id] for object_id in core_ids], labels),
    }
    output = args.audit / "team_metrics.md"
    output.write_text(make_markdown(core_ids, annotators, human_metrics, model_metrics), encoding="utf-8")
    print(f"Core objects: {len(core_ids)}; annotators: {', '.join(annotators)}")
    print(f"pairwise_agreement={human_metrics['pairwise_agreement']:.6f}")
    print(f"alpha={human_metrics['alpha']:.6f}")
    print(f"unanimous_share={human_metrics['unanimous_share']:.6f}")
    for name, result in model_metrics.items():
        print(f"{name}: {result['matches']}/{result['n']} agreement={result['agreement']:.6f} ci={result['ci']}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
