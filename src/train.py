"""Algorithm-agnostic training script for HalfCheetah (MuJoCo) + Stable-Baselines3.

All algorithm hyper-parameters live in YAML config files (see src/configs/).
CLI flags exist only for per-run overrides (env, seed, wandb, …).

Phases are encoded as compound env IDs: HalfCheetah-v5:run, HalfCheetah-v5:backflip,
HalfCheetah-v5:efficient. The correct reward wrapper is applied automatically.

Usage
-----
    python src/train.py -c src/configs/ppo.yaml
    python src/train.py -c src/configs/sac.yaml --env-id HalfCheetah-v5:backflip --wandb
    python src/train.py -c src/configs/td3.yaml --total-steps 500000 --wandb
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# ── Algorithm registry ───────────────────────────────────────────────────
ALGO_MAP: dict[str, type] = {
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
}


# ── Compound env_id parsing ─────────────────────────────────────────────
def parse_env_id(env_id: str) -> tuple[str, str | None]:
    """Split compound id like 'HalfCheetah-v5:backflip' into (base_id, phase)."""
    if ":" in env_id:
        base, phase = env_id.rsplit(":", 1)
        return base.strip(), phase.strip()
    return env_id, None


# ── Reward wrappers for HalfCheetah phases ───────────────────────────────
class BackflipReward(gym.RewardWrapper):
    """Reward = L2 norm of torso angular velocity (encourages flipping)."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # ── TODO (Phase 2): compute the backflip reward ──────────────
        # Hints:
        #   - Access joint velocities via: self.unwrapped.data.qvel
        #   - Indices 3:6 are the torso angular velocity (wx, wy, wz)
        #   - The reward should be the L2 norm of that angular velocity vector
        #   - Use np.array(..., dtype=np.float64) and np.linalg.norm(...)
        #   - Handle the edge case where qvel has fewer than 6 elements
        new_reward = 0.0  # ← replace this with your implementation
        # ─────────────────────────────────────────────────────────────
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


# ── Config helpers ───────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    """Load a YAML config file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> None:
    """Mutate *cfg* with any non-None CLI overrides."""
    env_cfg = cfg.setdefault("env", {})
    training = cfg.setdefault("training", {})
    wandb_cfg = cfg.setdefault("wandb", {})

    if args.env_id:
        env_cfg["id"] = args.env_id
    if args.total_steps is not None:
        training["total_steps"] = args.total_steps
    if args.seed is not None:
        training["seed"] = args.seed
    if args.model_path:
        training["model_path"] = args.model_path
    if args.wandb_project:
        wandb_cfg["project"] = args.wandb_project
    if args.wandb_entity:
        wandb_cfg["entity"] = args.wandb_entity
    if args.wandb_run_name:
        wandb_cfg["run_name"] = args.wandb_run_name
    if args.wandb_tags:
        wandb_cfg["tags"] = [t.strip() for t in args.wandb_tags.split(",")]


# ── Environment factory ─────────────────────────────────────────────────
def make_env(
    env_id: str,
    seed: int,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a HalfCheetah environment; apply phase reward wrapper if compound env_id."""
    base_id, phase = parse_env_id(env_id)
    env = gym.make(base_id, render_mode=render_mode)
    env = _wrap_phase(env, phase)
    env.reset(seed=seed)
    return env


# ── Video recording ─────────────────────────────────────────────────────
def record_episode_video(
    env_id: str,
    seed: int,
    model,
) -> tuple[str | None, str | None]:
    """Record a single evaluation episode to **mp4**.

    Returns (video_path, tmp_dir) on success, (None, None) on failure
    (e.g. headless environment without EGL).
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        env = make_env(env_id, seed, render_mode="rgb_array")
        rec = RecordVideo(
            env,
            tmp_dir,
            episode_trigger=lambda ep: ep == 0,
            disable_logger=True,
        )

        obs, _ = rec.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = rec.step(action)
            done = terminated or truncated
        rec.close()

        videos = sorted(
            Path(tmp_dir).glob("*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if videos:
            return str(videos[0]), tmp_dir
    except Exception as e:
        print(f"[video] Skipping video recording: {e}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return None, None


# ── W&B callbacks ────────────────────────────────────────────────────────
class EvalCallback(BaseCallback):
    """Periodic evaluation with optional mp4 video logging to W&B."""

    def __init__(self, eval_env: gym.Env, cfg: dict):
        super().__init__()
        self.eval_env = eval_env
        eval_ = cfg.get("evaluation", {})
        self.eval_freq: int = eval_.get("interval", 0)
        self.n_episodes: int = eval_.get("episodes", 5)
        self.video_freq: int = eval_.get("video_interval", 0)
        self.video_fps: int = eval_.get("video_fps", 10)
        self.env_id: str = cfg.get("env", {}).get("id", "")
        self.seed: int = cfg.get("training", {}).get("seed", 42)

    def _evaluate(self) -> dict[str, float]:
        returns: list[float] = []
        successes = 0
        for _ in range(self.n_episodes):
            obs, _ = self.eval_env.reset()
            done, total = False, 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = self.eval_env.step(action)
                done = term or trunc
                total += r
            returns.append(total)
            successes += int(total > 0)
        return {
            "eval/mean_reward": float(np.mean(returns)),
            "eval/std_reward": float(np.std(returns)),
            "eval/max_reward": float(np.max(returns)),
            "eval/success_rate": successes / max(len(returns), 1),
        }

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        import wandb

        metrics = self._evaluate()

        tmp_dir = None
        if self.video_freq > 0 and self.n_calls % self.video_freq == 0:
            path, tmp_dir = record_episode_video(
                self.env_id,
                self.seed + 100,
                self.model,
            )
            if path:
                metrics["eval/video"] = wandb.Video(
                    path, fps=self.video_fps, format="mp4",
                )

        wandb.log(metrics, step=self.num_timesteps)

        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return True


class TrainLogCallback(BaseCallback):
    """Streams rolling training statistics to W&B."""

    def __init__(self, log_interval: int = 1000):
        super().__init__()
        self.log_interval = log_interval
        self.rewards: deque[float] = deque(maxlen=100)
        self.lengths: deque[int] = deque(maxlen=100)
        self.successes: deque[int] = deque(maxlen=100)
        self.t0 = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self.rewards.append(ep.get("r", 0.0))
                self.lengths.append(ep.get("l", 0))
                s = info.get(
                    "is_success",
                    info.get("success", ep.get("r", 0) > 0),
                )
                self.successes.append(int(bool(s)))

        if (
            self.log_interval > 0
            and self.n_calls % self.log_interval == 0
            and self.rewards
        ):
            import wandb

            wandb.log(
                {
                    "train/reward_mean": float(np.mean(self.rewards)),
                    "train/length_mean": float(np.mean(self.lengths)),
                    "train/success_rate": float(np.mean(self.successes)),
                    "train/elapsed_sec": time.time() - self.t0,
                },
                step=self.num_timesteps,
            )
        return True


# ── Model builder ────────────────────────────────────────────────────────
def build_model(cfg: dict, env):
    """Instantiate an SB3 algorithm from a config dict."""
    algo_name = cfg["algorithm"]
    if algo_name not in ALGO_MAP:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. "
            f"Available: {', '.join(ALGO_MAP)}"
        )
    algo_cls = ALGO_MAP[algo_name]
    hp = dict(cfg.get("hyperparameters", {}))
    training = cfg.get("training", {})

    policy_kwargs: dict = {}
    if "net_arch" in hp:
        policy_kwargs["net_arch"] = hp.pop("net_arch")
    if "activation_fn" in hp:
        import torch.nn as nn

        _act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}
        name = hp.pop("activation_fn")
        if name not in _act_map:
            raise ValueError(f"Unknown activation_fn '{name}'. Use: {list(_act_map)}")
        policy_kwargs["activation_fn"] = _act_map[name]

    return algo_cls(
        cfg.get("policy", "MlpPolicy"),
        env,
        **hp,
        **({"policy_kwargs": policy_kwargs} if policy_kwargs else {}),
        verbose=training.get("verbose", 0),
        seed=training.get("seed", 42),
    )


# ── Programmatic entry point (e.g. for Colab) ─────────────────────────────
def run_training(cfg: dict, use_wandb: bool = False) -> Path:
    """Run training from a config dict. Returns the path to the saved model.

    Use this from notebooks or scripts when you have the config in memory
    (e.g. an editable dict). Same config structure as the YAML files.
    """
    env_cfg = cfg.get("env", {})
    training = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    wandb_cfg = cfg.get("wandb", {})

    env_id = env_cfg["id"]
    seed = training.get("seed", 42)
    total_steps = training.get("total_steps", 500_000)
    model_path = Path(training.get("model_path", "models/agent.zip"))
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if use_wandb:
        import wandb

        wandb.init(
            project=wandb_cfg.get("project", "aidl-rl-benchmark"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name"),
            tags=wandb_cfg.get("tags", ["training"]),
            config={
                "env_id": env_id,
                "algorithm": cfg["algorithm"],
                **cfg.get("hyperparameters", {}),
            },
        )

    vec_env = make_vec_env(
        lambda: make_env(env_id, seed),
        n_envs=training.get("n_envs", 1),
        seed=seed,
    )

    model = build_model(cfg, vec_env)

    callbacks: list[BaseCallback] = []
    eval_env = None

    if use_wandb:
        callbacks.append(TrainLogCallback(training.get("log_interval", 1000)))
        if eval_cfg.get("interval", 0) > 0:
            eval_env = make_env(env_id, seed + 1)
            callbacks.append(EvalCallback(eval_env, cfg))

    t0 = time.time()
    try:
        model.learn(total_timesteps=total_steps, callback=callbacks or None)
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving current model...")
    elapsed = time.time() - t0
    model.save(str(model_path))
    steps_done = model.num_timesteps
    print(f"Saved model to {model_path}  ({elapsed:.1f}s, {steps_done} steps)")

    if use_wandb:
        import wandb

        wandb.log({"train/total_sec": elapsed})
        wandb.finish()
    if eval_env:
        eval_env.close()

    return model_path


# ── CLI & main ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train an RL agent on HalfCheetah (config-driven)",
    )
    p.add_argument(
        "--config", "-c", required=True,
        help="Path to algorithm YAML config (e.g. src/configs/ppo.yaml)",
    )
    p.add_argument("--env-id", help="Override env.id from config")
    p.add_argument("--total-steps", type=int, help="Override training.total_steps")
    p.add_argument("--seed", type=int, help="Override training.seed")
    p.add_argument("--model-path", help="Override training.model_path")
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT"))
    p.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    p.add_argument("--wandb-run-name", default=os.getenv("WANDB_RUN_NAME"))
    p.add_argument("--wandb-tags", default=os.getenv("WANDB_TAGS"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    _apply_cli_overrides(cfg, args)
    run_training(cfg, use_wandb=args.wandb)


if __name__ == "__main__":
    main()
