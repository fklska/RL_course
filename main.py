import time
import random

import gymnasium as gym
from pathlib import Path
import json
import csv

from scripts.bfs import ENV_ID, solve, transition_baseline

def gen():
    # EDIT HERE — измените только значения справа от знака =
    TEAM_ID = "K01"                 # K01 ... K16
    DIRECTION = "code"              # code / math / labeling / environments / logs
    ANNOTATOR_IDS = ["A1","A2","A3"]  # фактически участвующие разметчики (2–4)
    PAIRWISE_TASK = False             # True, если инструкция просит сравнивать A/B
    AUDIT_DIR = "audit"              # папка с входными файлами, подготовленными командой

    Path(AUDIT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Рабочая папка: {Path(AUDIT_DIR).resolve()}")
    print("Проверьте, что команда положила сюда slice.jsonl и instruction.md.")
    create_annotations_template(AUDIT_DIR, ANNOTATOR_IDS, PAIRWISE_TASK)

    # LOCKED — генератор командных шаблонов

def create_annotations_template(audit_dir, annotator_ids, pairwise=False, seed=2026):
    root=Path(audit_dir)
    rows=[json.loads(x) for x in (root/"slice.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    core=[x["id"] for x in rows if x["group"]=="core"]
    second=[x["id"] for x in rows if x["group"]=="second"]
    if len(core)!=20 or len(second)!=32:
        raise ValueError("В предоставленном срезе должно быть 20 объектов core и 32 объекта second")
    if not 2 <= len(annotator_ids) <= 4:
        raise ValueError("Укажите от 2 до 4 фактически участвующих разметчиков")
    rng=random.Random(seed)
    rng.shuffle(second)
    out=[]
    for person in annotator_ids:
        for object_id in core:
            out.append({"id":object_id,"annotator":person,"label":"",
                        "order":rng.choice(["A-first","B-first"]) if pairwise else "NA",
                        "instruction_version":"1.0","elapsed_seconds":""})
    for j, person in enumerate(annotator_ids):
        for object_id in second[j*8:(j+1)*8]:
            out.append({"id":object_id,"annotator":person,"label":"",
                        "order":rng.choice(["A-first","B-first"]) if pairwise else "NA",
                        "instruction_version":"1.0","elapsed_seconds":""})
    path=root/"annotations.csv"
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["id","annotator","label","order","instruction_version","elapsed_seconds"])
        writer.writeheader(); writer.writerows(out)
    with (root/"baseline_predictions.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f); w.writerow(["id","verdict"])
        for x in rows: w.writerow([x["id"],""])
    with (root/"probe_results.csv").open("w",encoding="utf-8-sig",newline="") as f:
        csv.writer(f).writerow(["probe","n","metric","value","interpretation"])
    print("Созданы: annotations.csv, baseline_predictions.csv, probe_results.csv")
    print("Заполните метки и результаты по инструкции; исходные ID и заголовки оставьте без изменений.")
    return path



def main():
    n = [random.randint(0, 999999) for i in range(20)]

    for seed in n:
        env = gym.make(ENV_ID, render_mode="human")
        env.unwrapped.metadata = {**env.unwrapped.metadata, "render_fps": 30}
        env.reset(seed=seed)
        actions = solve(env, transition_baseline)
        print(f"seed={seed} steps={len(actions) if actions is not None else None}")

        for action in actions or []:
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        time.sleep(1)
        env.close()

gen()
#main()
