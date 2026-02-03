# Student Starter Kit

## 1) Install
```bash
pip install gymnasium minigrid stable-baselines3
```

## 2) Train a baseline
```bash
python src/train.py --env-id MiniGrid-Empty-6x6-v0 --algo ppo --total-steps 200000
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

## Notes
- Env 1 must reach success_rate ≥ 0.6 to unlock Env 2.
- Your total score is the weighted sum of best mean_return per environment.
