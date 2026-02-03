import argparse
from pathlib import Path

import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

ALGO_MAP = {
    "ppo": PPO,
    "dqn": DQN,
}


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env = FlatObsWrapper(env)
    env.reset(seed=seed)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--algo", default="ppo", choices=ALGO_MAP.keys())
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default="models/agent.zip")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(
        lambda: make_env(args.env_id, args.seed),
        n_envs=1,
        seed=args.seed
    )
    vec_env = VecMonitor(vec_env)

    algo_cls = ALGO_MAP[args.algo]
    model = algo_cls("MlpPolicy", vec_env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.total_steps)
    model.save(str(model_path))

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
