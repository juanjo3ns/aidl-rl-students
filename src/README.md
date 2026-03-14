# Student Starter Kit — HalfCheetah RL Competition

## 1) Install (Python 3.11+)
```bash
uv venv --python 3.13 .venv
uv pip install -p .venv/bin/python -r src/requirements.txt -r eval/requirements.txt
```

Requires `gymnasium[mujoco]` for HalfCheetah (MuJoCo).

## 2) Train (config-driven)
All hyperparameters live in YAML configs under `src/configs/`. Use `-c` to pick an algorithm and override env or steps as needed.

**Phase 1 — Run (default):**
```bash
python src/train.py -c src/configs/ppo.yaml
python src/train.py -c src/configs/sac.yaml --wandb
```

**Phase 2 — Backflip:** train with the backflip reward by overriding env-id:
```bash
python src/train.py -c src/configs/sac.yaml --env-id HalfCheetah-v5:backflip --wandb
```

**Phase 3 — Efficient Run:**
```bash
python src/train.py -c src/configs/sac.yaml --env-id HalfCheetah-v5:efficient --wandb
```

Override steps or seed:
```bash
python src/train.py -c src/configs/ppo.yaml --total-steps 300000 --seed 42
```

## 3) Evaluate + Submit
Use the same `env_id` (including phase) and algo as in your training config.

```bash
python src/eval_submit.py \
  --session-code YOUR_SESSION_CODE \
  --team-name "Team Turbo" \
  --env-id HalfCheetah-v5:run \
  --algo sac \
  --model-path models/sac_agent.zip \
  --api-url https://YOUR-DOMAIN/api/submit \
  --video-path /path/to/attempt.mp4
```

For Phase 2 or 3, set `--env-id HalfCheetah-v5:backflip` or `HalfCheetah-v5:efficient` and the model path for that phase.
`--video-path` can be a local file (or Colab path like `/content/attempt.mp4`) and is uploaded automatically to the Railway bucket before submitting the score.
If you already uploaded elsewhere, you can still use `--video-url` or `--video-key`.

### Upload to Railway Bucket (optional)
The script uses this API internally when `--video-path` is provided:
- Requests a presigned URL at `/api/video/upload-url`
- Uploads the video file to bucket via signed `PUT`
- Submits with returned `video_key`

If your Python environment fails TLS verification on submit, try:
```bash
python src/eval_submit.py ... --ca-bundle /path/to/ca-bundle.pem
```
As a last resort for local testing only:
```bash
python src/eval_submit.py ... --insecure-skip-verify
```

## Optional: Weights & Biases
Set `WANDB_API_KEY` and add `--wandb` to training and submission.
```bash
python src/train.py -c src/configs/sac.yaml --wandb
python src/eval_submit.py ... --wandb --eval-video --eval-video-format mp4
```

## Algorithms and configs
- **PPO** — `src/configs/ppo.yaml`
- **SAC** — `src/configs/sac.yaml` (recommended for continuous control)
- **TD3** — `src/configs/td3.yaml`
- **A2C** — `src/configs/a2c.yaml`

## Phase progression
- **Phase 1 (Run)** is always available. Reach the session’s mean_return threshold to unlock Phase 2.
- **Phase 2 (Backflip)** unlocks after Phase 1. Reach its threshold to unlock Phase 3.
- **Phase 3 (Efficient Run)** — your weighted total across phases determines your rank.

Your total score is the weighted sum of your best mean_return per phase.
