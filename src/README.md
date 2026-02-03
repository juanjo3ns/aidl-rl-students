# Student Starter Kit

## 1) Install (Python 3.13)
We recommend Python 3.13 to avoid building `pygame` from source.
```bash
uv venv --python 3.13 .venv
uv pip install -p .venv/bin/python -r src/requirements.txt
```

## 2) Train a baseline
```bash
python src/train.py --env-id MiniGrid-Empty-6x6-v0 --algo ppo --total-steps 200000 --verbose 0
```

## 3) Evaluate + Submit
```bash
python src/eval_submit.py \
  --session-code LAB2026 \
  --team-name "Team Turbo" \
  --env-id MiniGrid-Empty-6x6-v0 \
  --algo ppo \
  --model-path models/agent.zip \
  --api-url https://YOUR-VERCEL-DOMAIN/api/submit
```

## Optional: Weights & Biases logging
Set `WANDB_API_KEY` and add `--wandb` to log training and submissions.
```bash
python src/train.py --env-id MiniGrid-Empty-6x6-v0 --algo ppo --total-steps 200000 --wandb --log-interval 1000 --eval-interval 10000 --eval-video
python src/eval_submit.py --session-code YOUR_SESSION_CODE --team-name "Team Turbo" --env-id MiniGrid-Empty-6x6-v0 --algo ppo --model-path models/agent.zip --api-url https://YOUR-VERCEL-DOMAIN/api/submit --wandb --eval-video
```

## Notes
- Env 1 must reach success_rate ≥ 0.6 to unlock Env 2.
- Your total score is the weighted sum of best mean_return per environment.
