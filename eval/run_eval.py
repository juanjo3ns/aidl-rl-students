import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO, DQN, A2C

ALGO_MAP = {
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C,
}


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env = FlatObsWrapper(env)
    env.reset(seed=seed)
    return env


def evaluate(model, env_id: str, seeds, episodes_per_seed: int):
    episode_returns = []
    successes = 0
    total_episodes = 0

    for seed in seeds:
        for ep in range(episodes_per_seed):
            env = make_env(env_id, seed + ep)
            obs, _ = env.reset(seed=seed + ep)
            done = False
            total_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
            env.close()
            episode_returns.append(total_reward)
            if total_reward > 0:
                successes += 1
            total_episodes += 1

    mean_return = float(np.mean(episode_returns))
    std_return = float(np.std(episode_returns))
    max_return = float(np.max(episode_returns))
    success_rate = successes / max(total_episodes, 1)

    return {
        "mean_return": mean_return,
        "std_return": std_return,
        "max_return": max_return,
        "success_rate": success_rate,
        "num_episodes": total_episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--algo", required=True, choices=ALGO_MAP.keys())
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seeds", default=str(Path(__file__).with_name("seeds.json")))
    parser.add_argument("--episodes-per-seed", type=int, default=5)
    args = parser.parse_args()

    algo_cls = ALGO_MAP[args.algo]
    model = algo_cls.load(args.model_path)

    seeds = json.loads(Path(args.seeds).read_text())

    start = time.time()
    metrics = evaluate(model, args.env_id, seeds, args.episodes_per_seed)
    runtime_sec = time.time() - start

    result = {
        "env_id": args.env_id,
        "algo": args.algo,
        "runtime_sec": runtime_sec,
        **metrics,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
