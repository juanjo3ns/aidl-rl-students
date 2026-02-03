import argparse
import json
import subprocess
import sys
import urllib.request


def run_eval(env_id: str, algo: str, model_path: str, seeds_path: str, episodes_per_seed: int):
    cmd = [
        sys.executable,
        "-m",
        "eval.run_eval",
        "--env-id",
        env_id,
        "--algo",
        algo,
        "--model-path",
        model_path,
        "--seeds",
        seeds_path,
        "--episodes-per-seed",
        str(episodes_per_seed),
    ]
    output = subprocess.check_output(cmd, text=True)
    return json.loads(output)


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


if __name__ == "__main__":
    main()
