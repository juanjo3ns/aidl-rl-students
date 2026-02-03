import argparse
import os
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

ALGO_MAP = {
    "ppo": PPO,
    "dqn": DQN,
}


def make_env(env_id: str, seed: int, render_mode: str | None = None):
    env = gym.make(env_id, render_mode=render_mode)
    env = FlatObsWrapper(env)
    env.reset(seed=seed)
    return env


class WandbEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        record_video: bool,
        video_interval: int,
        video_fps: int,
        video_max_frames: int,
        video_format: str,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.record_video = record_video
        self.video_interval = video_interval
        self.video_fps = video_fps
        self.video_max_frames = video_max_frames
        self.video_format = video_format

    def _run_eval(self, record_video: bool):
        episode_returns = []
        successes = 0
        frames = []

        for ep in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
                if record_video and ep == 0 and len(frames) < self.video_max_frames:
                    frame = self.eval_env.render()
                    if frame is not None:
                        frames.append(frame)

            episode_returns.append(total_reward)
            if total_reward > 0:
                successes += 1

        metrics = {
            "eval/mean_reward": float(np.mean(episode_returns)),
            "eval/std_reward": float(np.std(episode_returns)),
            "eval/max_reward": float(np.max(episode_returns)),
            "eval/success_rate": successes / max(len(episode_returns), 1),
            "eval/episodes": len(episode_returns),
        }
        return metrics, frames

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.n_calls % self.eval_freq == 0:
            import wandb

            record = self.record_video and (self.video_interval > 0) and (self.n_calls % self.video_interval == 0)
            metrics, frames = self._run_eval(record_video=record)
            metrics["train/timesteps"] = self.num_timesteps

            if record and frames:
                video_frames = np.stack(frames).astype(np.uint8)
                video = wandb.Video(video_frames, fps=self.video_fps, format=self.video_format)
                metrics["eval/video"] = video

            wandb.log(metrics, step=self.num_timesteps)
        return True


class WandbTrainCallback(BaseCallback):
    def __init__(self, log_interval: int):
        super().__init__()
        self.log_interval = log_interval
        self.ep_rewards = deque(maxlen=100)
        self.ep_lengths = deque(maxlen=100)
        self.ep_successes = deque(maxlen=100)
        self.start_time = time.time()

    def _collect_logger_metrics(self):
        metrics = {}
        logger = getattr(self.model, "logger", None)
        name_to_value = getattr(logger, "name_to_value", None)
        if isinstance(name_to_value, dict):
            for key, value in name_to_value.items():
                if not isinstance(value, (int, float)):
                    continue
                if key.startswith("train/") or key.startswith("rollout/") or "loss" in key:
                    metrics[f"sb3/{key}"] = value
        return metrics

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                episode = info["episode"]
                self.ep_rewards.append(episode.get("r", 0.0))
                self.ep_lengths.append(episode.get("l", 0))
                success = False
                if "is_success" in info:
                    success = bool(info.get("is_success"))
                elif "success" in info:
                    success = bool(info.get("success"))
                else:
                    success = episode.get("r", 0.0) > 0.0
                self.ep_successes.append(1 if success else 0)

        if self.log_interval > 0 and self.n_calls % self.log_interval == 0:
            import wandb

            metrics = {
                "train/num_timesteps": self.num_timesteps,
                "train/elapsed_sec": time.time() - self.start_time,
            }
            if self.ep_rewards:
                metrics["train/episode_reward_mean"] = sum(self.ep_rewards) / len(self.ep_rewards)
            if self.ep_lengths:
                metrics["train/episode_length_mean"] = sum(self.ep_lengths) / len(self.ep_lengths)
            if self.ep_successes:
                metrics["train/success_count_window"] = sum(self.ep_successes)
                metrics["train/success_rate_window"] = sum(self.ep_successes) / len(self.ep_successes)
            metrics.update(self._collect_logger_metrics())
            wandb.log(metrics, step=self.num_timesteps)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--algo", default="ppo", choices=ALGO_MAP.keys())
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default="models/agent.zip")
    parser.add_argument("--verbose", type=int, default=0, help="SB3 verbosity (0=silent).")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "aidl-rl-benchmark"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-group", default=os.getenv("WANDB_GROUP"))
    parser.add_argument("--wandb-run-name", default=os.getenv("WANDB_RUN_NAME"))
    parser.add_argument("--wandb-tags", default=os.getenv("WANDB_TAGS", "training"))
    parser.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", "online"))
    parser.add_argument("--eval-interval", type=int, default=0, help="Timesteps between evals.")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=1000, help="Timesteps between train logs.")
    parser.add_argument("--eval-video", action="store_true", help="Upload eval video to W&B.")
    parser.add_argument("--eval-video-interval", type=int, default=0, help="Timesteps between eval videos.")
    parser.add_argument("--eval-video-fps", type=int, default=10)
    parser.add_argument("--eval-video-max-frames", type=int, default=500)
    parser.add_argument("--eval-video-format", default="gif", choices=["gif", "mp4"])
    parser.add_argument("--dqn-learning-rate", type=float, default=1e-4)
    parser.add_argument("--dqn-buffer-size", type=int, default=1_000_000)
    parser.add_argument("--dqn-learning-starts", type=int, default=50_000)
    parser.add_argument("--dqn-batch-size", type=int, default=32)
    parser.add_argument("--dqn-gamma", type=float, default=0.99)
    parser.add_argument("--dqn-tau", type=float, default=1.0)
    parser.add_argument("--dqn-train-freq", type=int, default=4)
    parser.add_argument("--dqn-gradient-steps", type=int, default=1)
    parser.add_argument("--dqn-target-update-interval", type=int, default=10_000)
    parser.add_argument("--dqn-exploration-fraction", type=float, default=0.1)
    parser.add_argument("--dqn-exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--dqn-exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--dqn-net-arch", default="")
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
    model_kwargs = {
        "verbose": args.verbose,
        "seed": args.seed,
    }
    if args.algo == "dqn":
        model_kwargs.update(
            {
                "learning_rate": args.dqn_learning_rate,
                "buffer_size": args.dqn_buffer_size,
                "learning_starts": args.dqn_learning_starts,
                "batch_size": args.dqn_batch_size,
                "gamma": args.dqn_gamma,
                "tau": args.dqn_tau,
                "train_freq": args.dqn_train_freq,
                "gradient_steps": args.dqn_gradient_steps,
                "target_update_interval": args.dqn_target_update_interval,
                "exploration_fraction": args.dqn_exploration_fraction,
                "exploration_initial_eps": args.dqn_exploration_initial_eps,
                "exploration_final_eps": args.dqn_exploration_final_eps,
            }
        )
        if args.dqn_net_arch:
            net_arch = [int(x.strip()) for x in args.dqn_net_arch.split(",") if x.strip()]
            if net_arch:
                model_kwargs["policy_kwargs"] = {"net_arch": net_arch}

    model = algo_cls("MlpPolicy", vec_env, **model_kwargs)

    callbacks = []
    eval_env = None
    if args.eval_interval > 0:
        eval_env = make_env(args.env_id, args.seed + 1, render_mode="rgb_array" if args.eval_video else None)
        if args.wandb:
            callbacks.append(
                WandbEvalCallback(
                    eval_env,
                    args.eval_interval,
                    args.eval_episodes,
                    args.eval_video,
                    args.eval_video_interval or args.eval_interval,
                    args.eval_video_fps,
                    args.eval_video_max_frames,
                    args.eval_video_format,
                )
            )
    if args.wandb:
        callbacks.append(WandbTrainCallback(args.log_interval))

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
