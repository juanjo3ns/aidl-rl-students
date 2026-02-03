import argparse
import json
import os
import urllib.request

import numpy as np

from eval.run_eval import ALGO_MAP, evaluate, make_env


def run_eval(env_id: str, algo: str, model_path: str, seeds_path: str, episodes_per_seed: int):
    model_cls = ALGO_MAP[algo]
    model = model_cls.load(model_path)
    with open(seeds_path, "r", encoding="utf-8") as f:
        seeds = json.load(f)
    return evaluate(model, env_id, seeds, episodes_per_seed)


def record_video(env_id: str, model_path: str, algo: str, max_frames: int):
    model_cls = ALGO_MAP[algo]
    model = model_cls.load(model_path)
    env = make_env(env_id, seed=0, render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    frames = []
    done = False
    while not done and len(frames) < max_frames:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frame = env.render()
        if frame is not None:
            frame = np.asarray(frame)
            if frame.ndim == 2:
                frame = np.stack([frame, frame, frame], axis=-1)
            if frame.ndim == 3 and frame.shape[-1] == 4:
                frame = frame[..., :3]
            if frame.ndim == 3 and frame.shape[-1] == 3:
                if frame.dtype != np.uint8:
                    max_val = float(frame.max()) if frame.size else 0.0
                    if max_val <= 1.0:
                        frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        frame = frame.clip(0, 255).astype(np.uint8)
                frames.append(frame)
    env.close()
    if frames:
        return np.stack(frames).astype(np.uint8)
    return None


def submit(api_url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-code", required=True)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seeds", default="eval/seeds.json")
    parser.add_argument("--episodes-per-seed", type=int, default=5)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "aidl-rl-benchmark"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-group", default=os.getenv("WANDB_GROUP"))
    parser.add_argument("--wandb-run-name", default=os.getenv("WANDB_RUN_NAME"))
    parser.add_argument("--wandb-tags", default=os.getenv("WANDB_TAGS", "submission"))
    parser.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", "online"))
    parser.add_argument("--eval-video", action="store_true", help="Upload an eval video to W&B.")
    parser.add_argument("--eval-video-fps", type=int, default=10)
    parser.add_argument("--eval-video-max-frames", type=int, default=500)
    parser.add_argument("--eval-video-format", default="gif", choices=["gif", "mp4"])
    args = parser.parse_args()

    metrics = run_eval(args.env_id, args.algo, args.model_path, args.seeds, args.episodes_per_seed)

    payload = {
        "session_code": args.session_code,
        "team_name": args.team_name,
        "env_id": args.env_id,
        "algo": args.algo,
        "seed": 0,
        "num_episodes": metrics["num_episodes"],
        "mean_return": metrics["mean_return"],
        "std_return": metrics["std_return"],
        "max_return": metrics["max_return"],
        "success_rate": metrics["success_rate"],
        "runtime_sec": metrics["runtime_sec"],
    }

    response = submit(args.api_url, payload)
    print(json.dumps(response, indent=2))

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
                "session_code": args.session_code,
                "team_name": args.team_name,
                "env_id": args.env_id,
                "algo": args.algo,
            },
        )

        rank = None
        leaderboard = response.get("leaderboard", [])
        for idx, entry in enumerate(leaderboard):
            if entry.get("team_name") == args.team_name:
                rank = idx + 1
                break

        wandb.log(
            {
                "eval/mean_return": metrics["mean_return"],
                "eval/std_return": metrics["std_return"],
                "eval/max_return": metrics["max_return"],
                "eval/success_rate": metrics["success_rate"],
                "submission/rank": rank,
            }
        )
        if args.eval_video:
            frames = record_video(args.env_id, args.model_path, args.algo, args.eval_video_max_frames)
            if frames is not None:
                wandb.log({"eval/video": wandb.Video(frames, fps=args.eval_video_fps, format=args.eval_video_format)})
        wandb.finish()


if __name__ == "__main__":
    main()
