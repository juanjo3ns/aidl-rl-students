import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

ALGO_MAP = {
    "ppo": PPO,
    "dqn": DQN,
}


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env = FlatObsWrapper(env)
    env.reset(seed=seed)
    return env


class WandbEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )
            import wandb

            wandb.log(
                {
                    "eval/mean_reward": mean_reward,
                    "eval/std_reward": std_reward,
                    "train/timesteps": self.num_timesteps,
                }
            )
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--algo", default="ppo", choices=ALGO_MAP.keys())
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default="models/agent.zip")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "aidl-rl-benchmark"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-group", default=os.getenv("WANDB_GROUP"))
    parser.add_argument("--wandb-run-name", default=os.getenv("WANDB_RUN_NAME"))
    parser.add_argument("--wandb-tags", default=os.getenv("WANDB_TAGS", "training"))
    parser.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", "online"))
    parser.add_argument("--eval-interval", type=int, default=0, help="Timesteps between evals.")
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()],
            mode=args.wandb_mode,
            config={
                "env_id": args.env_id,
                "algo": args.algo,
                "total_steps": args.total_steps,
                "seed": args.seed,
            },
        )

    vec_env = make_vec_env(
        lambda: make_env(args.env_id, args.seed),
        n_envs=1,
        seed=args.seed
    )
    vec_env = VecMonitor(vec_env)

    algo_cls = ALGO_MAP[args.algo]
    model = algo_cls("MlpPolicy", vec_env, verbose=1, seed=args.seed)

    callbacks = []
    eval_env = None
    if args.eval_interval > 0:
        eval_env = make_env(args.env_id, args.seed + 1)
        if args.wandb:
            callbacks.append(WandbEvalCallback(eval_env, args.eval_interval, args.eval_episodes))

    start = time.time()
    model.learn(total_timesteps=args.total_steps, callback=callbacks if callbacks else None)
    elapsed = time.time() - start
    model.save(str(model_path))

    if args.wandb:
        import wandb

        wandb.log(
            {
                "train/elapsed_sec": elapsed,
                "train/total_timesteps": args.total_steps,
            }
        )
        wandb.finish()

    if eval_env is not None:
        eval_env.close()

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
