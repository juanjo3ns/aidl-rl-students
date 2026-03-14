# Student Bundle — HalfCheetah RL Competition

This bundle contains everything students need to train, evaluate, and submit agents for the 3-phase HalfCheetah competition.

## Install (Python 3.11+)
```bash
uv venv --python 3.13 .venv
uv pip install -p .venv/bin/python -r eval/requirements.txt -r src/requirements.txt
```

## Train (config-driven)
```bash
python src/train.py -c src/configs/sac.yaml --env-id HalfCheetah-v5:run --wandb
```
Use `--env-id HalfCheetah-v5:backflip` or `HalfCheetah-v5:efficient` for Phase 2 or 3. Configs: ppo, sac, td3, a2c in `src/configs/`.

## Evaluate + Submit
```bash
python src/eval_submit.py \
  --session-code YOUR_SESSION_CODE \
  --team-name "Team Turbo" \
  --env-id HalfCheetah-v5:run \
  --algo sac \
  --model-path models/sac_agent.zip \
  --api-url https://YOUR-DOMAIN/api/submit
```

## Optional: Weights & Biases
Set `WANDB_API_KEY` and add `--wandb` to training and submission.
```bash
python src/train.py -c src/configs/sac.yaml --wandb
python src/eval_submit.py ... --wandb --eval-video --eval-video-format mp4
```

## Notes
- Phase 1 (Run) is always available. Reach the mean_return threshold to unlock Phase 2 (Backflip), then Phase 3 (Efficient).
- Your total score is the weighted sum of best mean_return per phase.
