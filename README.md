# Student Bundle

This bundle contains everything students need to train, evaluate, and submit agents.

## Install
```bash
pip install -r eval/requirements.txt -r src/requirements.txt
```

## Train
```bash
python src/train.py --env-id MiniGrid-Empty-6x6-v0 --algo ppo --total-steps 200000
```

## Evaluate + Submit
```bash
python src/eval_submit.py \
  --session-code YOUR_SESSION_CODE \
  --team-name "Team Turbo" \
  --env-id MiniGrid-Empty-6x6-v0 \
  --algo ppo \
  --model-path models/agent.zip \
  --api-url https://YOUR-VERCEL-DOMAIN/api/submit
```

## Notes
- Env 1 must reach success_rate ≥ 0.6 to unlock Env 2.
- Your total score is the weighted sum of best mean_return per environment.
