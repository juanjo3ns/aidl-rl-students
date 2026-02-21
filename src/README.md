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
python src/train.py -c src/configs/sac.yaml --env-id HalfCheetah-v4:backflip --wandb
```

**Phase 3 — Efficient Run:**
```bash
python src/train.py -c src/configs/sac.yaml --env-id HalfCheetah-v4:efficient --wandb
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
  --env-id HalfCheetah-v4:run \
  --algo sac \
  --model-path models/sac_agent.zip \
  --api-url https://YOUR-DOMAIN/api/submit
```

For Phase 2 or 3, set `--env-id HalfCheetah-v4:backflip` or `HalfCheetah-v4:efficient` and the model path for that phase.

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
