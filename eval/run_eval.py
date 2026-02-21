import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, PPO, SAC, TD3

ALGO_MAP = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
}


def parse_env_id(env_id: str) -> tuple[str, str | None]:
    """Split compound id like 'HalfCheetah-v4:backflip' into (base_id, phase)."""
    if ":" in env_id:
        base, phase = env_id.rsplit(":", 1)
        return base.strip(), phase.strip()
    return env_id, None


class BackflipReward(gym.RewardWrapper):
    """Reward = L2 norm of torso angular velocity."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        qvel = self.unwrapped.data.qvel
        if len(qvel) >= 6:
            angular = np.array(qvel[3:6], dtype=np.float64)
            new_reward = float(np.linalg.norm(angular))
        else:
            new_reward = 0.0
        return obs, new_reward, terminated, truncated, info


class EfficientRunReward(gym.RewardWrapper):
    """Reward = forward_velocity / (1 + control_energy)."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        qvel = self.unwrapped.data.qvel
        forward_vel = float(qvel[0]) if len(qvel) > 0 else 0.0
        energy = float(np.square(np.asarray(action)).sum())
        new_reward = forward_vel / (1.0 + energy)
        return obs, new_reward, terminated, truncated, info


def _wrap_phase(env: gym.Env, phase: str | None) -> gym.Env:
    if phase == "backflip":
        return BackflipReward(env)
    if phase == "efficient":
        return EfficientRunReward(env)
    return env


def make_env(env_id: str, seed: int, render_mode: str | None = None) -> gym.Env:
    """Create HalfCheetah env; apply phase reward wrapper if compound env_id."""
    base_id, phase = parse_env_id(env_id)
    env = gym.make(base_id, render_mode=render_mode)
    env = _wrap_phase(env, phase)
    env.reset(seed=seed)
    return env


def evaluate(model, env_id: str, seeds, episodes_per_seed: int):
    start = time.time()
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

    runtime_sec = time.time() - start
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
        "runtime_sec": runtime_sec,
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

    metrics = evaluate(model, args.env_id, seeds, args.episodes_per_seed)

    result = {
        "env_id": args.env_id,
        "algo": args.algo,
        **metrics,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
