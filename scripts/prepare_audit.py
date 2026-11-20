"""Prepare fixed trajectories and human-readable cards without scorer verdicts."""

import argparse
from html import escape
import json
from math import atan2, cos, sin
from pathlib import Path

import gymnasium as gym
from PIL import Image, ImageDraw

from scripts.bfs import DIRECTIONS, ENV_ID, FORWARD, LEFT, RIGHT, read_world, solve, transition_baseline
from scripts.generate_configurations import verify_records, write_records

ACTION_NAMES = ("left", "right", "forward", "pickup", "drop", "toggle")
ACTION_RU = ("Поворот налево", "Поворот направо", "Шаг вперёд", "Взять предмет", "Положить предмет", "Открыть/закрыть дверь")
DIRECTION_RU = ("вправо (восток)", "вниз (юг)", "влево (запад)", "вверх (север)")
COLOR_RU = {"red": "красный", "green": "зелёный", "blue": "синий", "purple": "фиолетовый", "yellow": "жёлтый", "grey": "серый"}
TYPE_RU = {"key": "ключ", "ball": "мяч", "door": "дверь"}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def make_actions(env):
    """Use the complete baseline BFS plan for every audit configuration."""
    actions = solve(env, transition_baseline)
    if not actions:
        raise RuntimeError("Baseline planner did not produce a nonempty plan; do not silently replace cases")
    return actions


def describe_initial_state(env):
    world = env.unwrapped
    lines = [
        "Цель: взять мяч в руки.",
        "Координаты (x, y): x растёт вправо, y растёт вниз; верхняя левая клетка (0, 0).",
        f"Размер карты: {world.width} x {world.height}.",
        f"Агент: {tuple(int(x) for x in world.agent_pos)}, смотрит {DIRECTION_RU[world.agent_dir]}.",
        "В руках: ничего.",
        f"Лимит: {world.max_steps} действий с начала эпизода; счётчик сейчас 0.",
        "Предметы и двери:",
    ]
    for y in range(world.height):
        for x in range(world.width):
            obj = world.grid.get(x, y)
            if obj is None or obj.type == "wall":
                continue
            description = f"({x}, {y}): {TYPE_RU.get(obj.type, obj.type)}, цвет {COLOR_RU.get(obj.color, obj.color)}"
            if obj.type == "door":
                description += ", " + ("заперта" if obj.is_locked else "открыта" if obj.is_open else "закрыта, не заперта")
            lines.append(description)
    return "\n".join(lines)


def ascii_map(env):
    world = env.unwrapped
    symbols = {"wall": "#", "door": "D", "key": "K", "ball": "B"}
    lines = ["Карта: # стена, . пусто, D дверь, K ключ, B мяч, >v<^ агент."]
    lines.append("   " + " ".join(f"{x:02d}" for x in range(world.width)))
    for y in range(world.height):
        cells = []
        for x in range(world.width):
            obj = world.grid.get(x, y)
            cell = "." if obj is None else symbols.get(obj.type, "?")
            if (x, y) == tuple(world.agent_pos):
                cell = ">v<^"[world.agent_dir]
            cells.append(cell)
        lines.append(f"{y:02d} " + " ".join(f" {cell}" for cell in cells))
    return "\n".join(lines)


def case_content(env, actions):
    description = describe_initial_state(env) + "\n\n" + ascii_map(env)
    steps = "\n".join(f"{i}. {ACTION_NAMES[a]} — {ACTION_RU[a]}" for i, a in enumerate(actions, start=1))
    return description + "\n\nПоследовательность действий:\n" + steps + "\n\nВопрос: будет ли мяч взят до конца этой последовательности и до тайм-аута? Метка: success или failure."


def predicted_states(env, actions):
    walls, door_positions, door_colors, state = read_world(env)
    states = [state]
    for action in actions:
        state, success = transition_baseline(state, action, walls, door_positions, door_colors)
        states.append(state)
        if success:
            break
    return states


def draw_arrow(draw, start, end, fill, width=3):
    draw.line((start, end), fill=fill, width=width)
    angle = atan2(end[1] - start[1], end[0] - start[0])
    size = 7
    left = (end[0] - size * cos(angle - 0.55), end[1] - size * sin(angle - 0.55))
    right = (end[0] - size * cos(angle + 0.55), end[1] - size * sin(angle + 0.55))
    draw.polygon((end, left, right), fill=fill)


def map_image(env, tile_size=40, actions=None):
    image = Image.fromarray(env.unwrapped.get_frame(highlight=False, tile_size=tile_size))
    margin = 28
    canvas = Image.new("RGB", (image.width + margin, image.height + margin), "white")
    canvas.paste(image, (margin, margin))
    if actions is not None:
        states = predicted_states(env, actions)
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        def center(position):
            return (margin + int(position[0]) * tile_size + tile_size // 2,
                    margin + int(position[1]) * tile_size + tile_size // 2)

        for action, before, after in zip(actions, states, states[1:]):
            if action == FORWARD and before[0] != after[0]:
                draw_arrow(draw, center(before[0]), center(after[0]), (0, 190, 255, 220), 4)
            elif action in (LEFT, RIGHT):
                x, y = center(before[0])
                old_dx, old_dy = DIRECTIONS[before[1]]
                new_dx, new_dy = DIRECTIONS[after[1]]
                start = (x + old_dx * 13, y + old_dy * 13)
                end = (x + new_dx * 13, y + new_dy * 13)
                draw_arrow(draw, start, end, (255, 170, 0, 240), 2)
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for x in range(env.unwrapped.width):
        draw.text((margin + x * tile_size + 14, 5), str(x), fill="black")
    for y in range(env.unwrapped.height):
        draw.text((6, margin + y * tile_size + 13), str(y), fill="black")
    return canvas


STYLE = """body{font:17px/1.55 system-ui,sans-serif;color:#182332;background:#f5f7fa;margin:0}main{max-width:1200px;margin:auto;padding:28px}section{background:white;padding:24px;border-radius:12px;margin:18px 0}img{max-width:100%;height:auto}pre{white-space:pre-wrap;font:15px/1.65 ui-monospace,Consolas,monospace}a{color:#1557a0}nav{display:flex;gap:20px;flex-wrap:wrap}.cases{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px}li{padding:4px}h1{line-height:1.2}code{background:#eef2f6;padding:2px 5px}.case-layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);gap:28px;align-items:start}.map-panel{position:sticky;top:16px}.actions{margin:0;padding-left:30px}.actions li{padding:4px 0}@media(max-width:900px){main{padding:16px}.case-layout{grid-template-columns:1fr}.map-panel{position:static}}"""


def html_document(title, body):
    return f'<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>{escape(title)}</title><style>{STYLE}</style><main>{body}</main></html>\n'


def case_html(record, previous_id=None, next_id=None):
    nav = '<a href="index.html">Все объекты</a> <a href="../instruction.md">Инструкция</a>'
    if previous_id:
        nav += f' <a href="{previous_id}.html">Предыдущий</a>'
    if next_id:
        nav += f' <a href="{next_id}.html">Следующий</a>'
    description, remainder = record["content"].split("\n\nПоследовательность действий:\n", 1)
    _, question = remainder.split("\n\nВопрос:", 1)
    actions = ''.join(
        f'<li><code>{ACTION_NAMES[action]}</code> — {ACTION_RU[action]}</li>'
        for action in record["actions"]
    )
    body = f'<nav>{nav}</nav><h1>{record["id"]}</h1><p>Группа: {record["group"]}. Оцените только указанную последовательность действий.</p>'
    body += f'<section><div class="case-layout"><div class="map-panel"><h2>Карта и прогноз BFS</h2><img src="maps/{record["id"]}.png" alt="Начальная карта с координатами и предсказанной BFS траекторией."><p>Красный треугольник — агент, его вершина показывает направление взгляда. Цвета дверей и предметов уточнены ниже.</p><p>Голубые линии показывают перемещения, оранжевые стрелки — повороты. Это только прогноз упрощённого BFS по указанным действиям, а не фактическое выполнение среды; прогноз может быть ошибочным.</p></div><div><h2>Последовательность действий</h2><ol class="actions">{actions}</ol></div></div></section>'
    body += f'<section><h2>Условие</h2><pre>{escape(description)}</pre><p>Вопрос:{escape(question)}</p></section>'
    body += '<section><h2>Как записать ответ</h2><p>В своей строке annotations.csv укажите <code>success</code> или <code>failure</code> и время в секундах. Страница не сохраняет ответы. Не обсуждайте метку с другими до сохранения своего суждения.</p><p>Оценивайте по настоящим правилам MiniGrid из инструкции. Не запускайте модели и не ищите машинные вердикты.</p></section>'
    return html_document(record["id"], body)


def index_html(records):
    body = '<h1>Разметка K04</h1><p>52 объекта: 20 core и 32 second. Все участники размечают core независимо; дополнительные объекты — по назначению из учебного ноутбука.</p><p>Перед началом прочитайте <a href="../instruction.md">инструкцию версии 1.0</a>. Первые 10 объектов ядра — пилот. Эта страница не хранит ответы.</p>'
    for group in ("core", "second"):
        body += f'<section><h2>{group}</h2><div class="cases">'
        body += ''.join(f'<a href="{r["id"]}.html">{r["id"]}</a>' for r in records if r["group"] == group)
        body += '</div></section>'
    return html_document("K04 — объекты разметки", body)


def write_text_once(path, content):
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite changed material: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(content)


def prepare(configurations_path, audit_dir):
    configurations = read_jsonl(configurations_path)
    verify_records(configurations)
    records, images = [], []
    for config in configurations:
        env = gym.make(ENV_ID)
        try:
            env.reset(seed=config["seed"])
            actions = make_actions(env)
            records.append({
                "id": config["id"], "group": config["group"],
                "env_id": config["env_id"], "seed": config["seed"],
                "configuration_fingerprint": config["fingerprint"],
                "versions": config["versions"],
                "initial_state": config["initial_state"],
                "actions": actions,
                "content": case_content(env, actions),
                "materials": {"page": f"materials/{config['id']}.html", "map": f"materials/maps/{config['id']}.png"},
            })
            images.append(map_image(env, actions=actions))
        finally:
            env.close()
    # Prepare all plans before writing. No evaluator or real action replay here.
    write_records(audit_dir / "slice.jsonl", records)
    materials = audit_dir / "materials"
    for index, (record, image) in enumerate(zip(records, images)):
        previous_id = records[index - 1]["id"] if index else None
        next_id = records[index + 1]["id"] if index + 1 < len(records) else None
        write_text_once(materials / f"{record['id']}.html", case_html(record, previous_id, next_id))
        path = materials / "maps" / f"{record['id']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with Image.open(path) as existing:
                if existing.convert("RGB").size != image.size or existing.convert("RGB").tobytes() != image.tobytes():
                    raise FileExistsError(f"Refusing to overwrite changed map: {path}")
        else:
            image.save(path)
    write_text_once(materials / "index.html", index_html(records))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configurations", type=Path, default=Path("data/configurations.jsonl"))
    parser.add_argument("--audit-dir", type=Path, default=Path("audit"))
    args = parser.parse_args()
    records = prepare(args.configurations, args.audit_dir)
    print(f"Prepared {len(records)} fixed trajectories and human cards; no machine verdicts exported")


if __name__ == "__main__":
    main()
