import time
import random

import gymnasium as gym

from scripts.bfs import ENV_ID, solve, transition_baseline

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