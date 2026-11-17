import argparse

import gymnasium as gym
from PIL import Image

from bfs import ENV_ID, solve


def play(seed, fps=30, out=None):
    env = gym.make(ENV_ID, render_mode="rgb_array" if out else "human")
    env.unwrapped.metadata = {**env.unwrapped.metadata, "render_fps": fps}
    env.reset(seed=seed)
    actions = solve(env)
    frames = [env.render()] if out else []
    for action in actions:
        _, _, terminated, truncated, _ = env.step(action)
        if out:
            frames.append(env.render())

        if terminated or truncated:
            break

    if out:
        frames += [frames[-1]] * fps
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(out, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0)
    else:
        for _ in range(fps):
            env.render()

    env.close()
    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out")
    args = parser.parse_args()
    actions = play(args.seed, args.fps, args.out)
    print(f"seed={args.seed} steps={len(actions)}")


if __name__ == "__main__":
    main()
