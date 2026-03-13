import argparse
import json
import mimetypes
import os
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Ensure repo root is on sys.path so `eval` package is importable
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from eval.run_eval import ALGO_MAP, evaluate, make_env

# Algo choices for HalfCheetah (continuous control: ppo, a2c, sac, td3)
ALGO_CHOICES = list(ALGO_MAP.keys())


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


def _build_context_chain(
    *,
    ca_bundle: str | None,
    insecure_skip_verify: bool,
) -> list[ssl.SSLContext]:
    if insecure_skip_verify:
        return [ssl._create_unverified_context()]
    if ca_bundle:
        return [ssl.create_default_context(cafile=ca_bundle)]

    contexts = [ssl.create_default_context()]
    try:
        import truststore
    except Exception:
        truststore = None
    if truststore is not None:
        contexts.append(truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))

    try:
        import certifi
    except Exception:
        certifi = None
    if certifi is not None:
        contexts.append(ssl.create_default_context(cafile=certifi.where()))

    return contexts


def _http_error_message(error: urllib.error.HTTPError) -> str:
    body = error.read().decode("utf-8") if error.fp else ""
    try:
        parsed = json.loads(body)
        return str(parsed.get("error", body or error.reason))
    except Exception:
        return body or str(error.reason)


def _post_json_with_context(api_url: str, payload: dict, *, context: ssl.SSLContext, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, context=context, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _request_json(
    api_url: str,
    payload: dict,
    *,
    label: str,
    ca_bundle: str | None = None,
    insecure_skip_verify: bool = False,
    timeout: float = 30.0,
) -> dict:
    ssl_error: urllib.error.URLError | None = None
    for context in _build_context_chain(
        ca_bundle=ca_bundle,
        insecure_skip_verify=insecure_skip_verify,
    ):
        try:
            return _post_json_with_context(api_url, payload, context=context, timeout=timeout)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{label} failed ({e.code}): {_http_error_message(e)}") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, ssl.SSLCertVerificationError):
                ssl_error = e
                continue
            raise RuntimeError(f"{label} failed: {e.reason}") from e

    raise RuntimeError(
        "TLS certificate verification failed. "
        "Try --ca-bundle <path-to-ca.pem> or --insecure-skip-verify only for local testing."
    ) from ssl_error


def _upload_bytes_with_context(
    url: str,
    data: bytes,
    *,
    content_type: str,
    context: ssl.SSLContext,
    timeout: float,
):
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": content_type},
        method="PUT",
    )
    with urllib.request.urlopen(req, context=context, timeout=timeout):
        return


def upload_file_to_signed_url(
    upload_url: str,
    *,
    file_path: str,
    content_type: str,
    ca_bundle: str | None = None,
    insecure_skip_verify: bool = False,
    timeout: float = 30.0,
):
    path = Path(file_path).expanduser()
    if not path.exists():
        raise RuntimeError(f"Video file not found: {path}")
    if not path.is_file():
        raise RuntimeError(f"Video path is not a file: {path}")

    data = path.read_bytes()
    ssl_error: urllib.error.URLError | None = None

    for context in _build_context_chain(
        ca_bundle=ca_bundle,
        insecure_skip_verify=insecure_skip_verify,
    ):
        try:
            _upload_bytes_with_context(
                upload_url,
                data,
                content_type=content_type,
                context=context,
                timeout=timeout,
            )
            return
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Video upload failed ({e.code}): {_http_error_message(e)}") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, ssl.SSLCertVerificationError):
                ssl_error = e
                continue
            raise RuntimeError(f"Video upload failed: {e.reason}") from e

    raise RuntimeError(
        "TLS certificate verification failed during video upload. "
        "Try --ca-bundle <path-to-ca.pem> or --insecure-skip-verify only for local testing."
    ) from ssl_error


def submit(
    api_url: str,
    payload: dict,
    *,
    ca_bundle: str | None = None,
    insecure_skip_verify: bool = False,
    timeout: float = 30.0,
) -> dict:
    return _request_json(
        api_url,
        payload,
        label="Submit",
        ca_bundle=ca_bundle,
        insecure_skip_verify=insecure_skip_verify,
        timeout=timeout,
    )


def derive_upload_url_from_submit_url(api_url: str) -> str:
    parsed = urllib.parse.urlsplit(api_url)
    path = parsed.path or ""
    if path.endswith("/api/submit"):
        upload_path = path[: -len("/api/submit")] + "/api/video/upload-url"
    elif path.endswith("/submit"):
        upload_path = path[: -len("/submit")] + "/video/upload-url"
    else:
        upload_path = "/api/video/upload-url"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, upload_path, "", ""))


def guess_video_content_type(video_path: str) -> str:
    guessed, _ = mimetypes.guess_type(video_path)
    if guessed and guessed.startswith("video/"):
        return guessed
    return "video/mp4"


def upload_video_and_get_key(
    *,
    api_url: str,
    session_code: str,
    team_name: str,
    env_id: str,
    video_path: str,
    video_content_type: str | None = None,
    video_upload_url: str | None = None,
    ca_bundle: str | None = None,
    insecure_skip_verify: bool = False,
    timeout: float = 30.0,
) -> str:
    path = Path(video_path).expanduser()
    if not path.exists():
        raise RuntimeError(f"Video file not found: {path}")
    if not path.is_file():
        raise RuntimeError(f"Video path is not a file: {path}")

    content_type = video_content_type or guess_video_content_type(str(path))
    upload_api_url = video_upload_url or derive_upload_url_from_submit_url(api_url)
    ticket = _request_json(
        upload_api_url,
        {
            "session_code": session_code,
            "team_name": team_name,
            "env_id": env_id,
            "file_name": path.name,
            "content_type": content_type,
        },
        label="Video upload-url request",
        ca_bundle=ca_bundle,
        insecure_skip_verify=insecure_skip_verify,
        timeout=timeout,
    )

    upload_url = ticket.get("upload_url")
    video_key = ticket.get("video_key")
    if not isinstance(upload_url, str) or not upload_url:
        raise RuntimeError("Video upload-url response missing upload_url.")
    if not isinstance(video_key, str) or not video_key:
        raise RuntimeError("Video upload-url response missing video_key.")

    upload_file_to_signed_url(
        upload_url,
        file_path=str(path),
        content_type=content_type,
        ca_bundle=ca_bundle,
        insecure_skip_verify=insecure_skip_verify,
        timeout=timeout,
    )
    return video_key


def evaluate_and_submit(
    session_code: str,
    team_name: str,
    env_id: str,
    algo: str,
    model_path: str,
    api_url: str,
    *,
    seeds_path: str = "eval/seeds.json",
    episodes_per_seed: int = 5,
    use_wandb: bool = False,
    eval_video: bool = False,
    video_fps: int = 10,
    video_max_frames: int = 500,
    video_format: str = "mp4",
    video_url: str | None = None,
    video_key: str | None = None,
    video_path: str | None = None,
    video_content_type: str | None = None,
    video_upload_url: str | None = None,
    ca_bundle: str | None = None,
    insecure_skip_verify: bool = False,
    timeout: float = 30.0,
) -> dict:
    """Evaluate the model and submit to the leaderboard API. Returns the API response.

    Use this from notebooks or scripts when you have paths and options in memory.
    """
    metrics = run_eval(env_id, algo, model_path, seeds_path, episodes_per_seed)
    if video_path and (video_url or video_key):
        raise ValueError("Use either video_path or video_url/video_key, not both.")

    resolved_video_key = video_key
    resolved_video_url = video_url

    if video_path:
        print(f"Uploading video from {video_path} ...")
        resolved_video_key = upload_video_and_get_key(
            api_url=api_url,
            session_code=session_code,
            team_name=team_name,
            env_id=env_id,
            video_path=video_path,
            video_content_type=video_content_type,
            video_upload_url=video_upload_url,
            ca_bundle=ca_bundle,
            insecure_skip_verify=insecure_skip_verify,
            timeout=timeout,
        )
        resolved_video_url = None
        print(f"Video uploaded. key={resolved_video_key}")

    payload = {
        "session_code": session_code,
        "team_name": team_name,
        "env_id": env_id,
        "algo": algo,
        "seed": 0,
        "num_episodes": metrics["num_episodes"],
        "mean_return": metrics["mean_return"],
        "std_return": metrics["std_return"],
        "max_return": metrics["max_return"],
        "success_rate": metrics["success_rate"],
        "runtime_sec": metrics["runtime_sec"],
        "video_url": resolved_video_url,
        "video_key": resolved_video_key,
    }
    response = submit(
        api_url,
        payload,
        ca_bundle=ca_bundle,
        insecure_skip_verify=insecure_skip_verify,
        timeout=timeout,
    )
    print(json.dumps(response, indent=2))

    if use_wandb:
        import wandb

        wandb.init(
            project=os.getenv("WANDB_PROJECT", "aidl-rl-benchmark"),
            entity=os.getenv("WANDB_ENTITY"),
            group=os.getenv("WANDB_GROUP"),
            name=os.getenv("WANDB_RUN_NAME"),
            tags=[t.strip() for t in (os.getenv("WANDB_TAGS", "submission") or "").split(",") if t.strip()],
            mode=os.getenv("WANDB_MODE", "online"),
            config={
                "session_code": session_code,
                "team_name": team_name,
                "env_id": env_id,
                "algo": algo,
            },
        )
        rank = None
        leaderboard = response.get("leaderboard", [])
        for idx, entry in enumerate(leaderboard):
            if entry.get("team_name") == team_name:
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
        if eval_video:
            frames = record_video(env_id, model_path, algo, video_max_frames)
            if frames is not None:
                wandb.log({"eval/video": wandb.Video(frames, fps=video_fps, format=video_format)})
        wandb.finish()

    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-code", required=True)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--algo", required=True, choices=ALGO_CHOICES)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seeds", default="eval/seeds.json")
    parser.add_argument("--episodes-per-seed", type=int, default=5)
    parser.add_argument("--api-url", required=True)
    parser.add_argument(
        "--ca-bundle",
        default=None,
        help="Path to a PEM bundle for HTTPS certificate verification.",
    )
    parser.add_argument(
        "--insecure-skip-verify",
        action="store_true",
        help="Disable HTTPS certificate verification (unsafe; only for local testing).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds for submission requests.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--eval-video", action="store_true", help="Upload an eval video to W&B.")
    parser.add_argument("--eval-video-fps", type=int, default=10)
    parser.add_argument("--eval-video-max-frames", type=int, default=500)
    parser.add_argument("--eval-video-format", default="mp4", choices=["gif", "mp4"])
    parser.add_argument(
        "--video-url",
        default=None,
        help="Public URL of the attempt video to attach in the leaderboard.",
    )
    parser.add_argument(
        "--video-key",
        default=None,
        help="Object key of a previously uploaded bucket video.",
    )
    parser.add_argument(
        "--video-path",
        default=None,
        help="Local video file path to upload directly to the leaderboard bucket (supports Colab paths like /content/...).",
    )
    parser.add_argument(
        "--video-content-type",
        default=None,
        help="Optional MIME type for --video-path upload (defaults from file extension).",
    )
    parser.add_argument(
        "--video-upload-url",
        default=None,
        help="Optional override for upload-url endpoint; defaults to /api/video/upload-url derived from --api-url.",
    )
    args = parser.parse_args()

    evaluate_and_submit(
        args.session_code,
        args.team_name,
        args.env_id,
        args.algo,
        args.model_path,
        args.api_url,
        seeds_path=args.seeds,
        episodes_per_seed=args.episodes_per_seed,
        use_wandb=args.wandb,
        eval_video=args.eval_video,
        video_fps=args.eval_video_fps,
        video_max_frames=args.eval_video_max_frames,
        video_format=args.eval_video_format,
        video_url=args.video_url,
        video_key=args.video_key,
        video_path=args.video_path,
        video_content_type=args.video_content_type,
        video_upload_url=args.video_upload_url,
        ca_bundle=args.ca_bundle,
        insecure_skip_verify=args.insecure_skip_verify,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
