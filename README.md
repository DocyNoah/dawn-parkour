# DAWN: Noise-Robust Quadruped Parkour

Official implementation of *DAWN: Noise-Robust Quadruped Parkour via Depth-Denoising World Models* (IROS 2026).
The code trains a Go1 parkour policy and replays checkpoints in Isaac Sim.

## Requirements

- Ubuntu 22.04
- NVIDIA GPU with CUDA 12.8+ support
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Setup

See [INSTALL.md](INSTALL.md).

## Train

We used an NVIDIA GeForce RTX 5090 for training.

```bash
source .venv/bin/activate
python -m dawn.train dawn
```

The default run trains 4,096 environments. Logs and checkpoints are saved in
`logs/DAWN/<run>/`; each `model_*` directory is a checkpoint.
To monitor training, run:

```bash
source .venv/bin/activate
tensorboard --logdir logs/DAWN
```

## Replay a checkpoint

Choose a `model_*` directory from a training run and replace the example path:

```bash
source .venv/bin/activate
python -m dawn.play \
  --checkpoint.model-dir "logs/DAWN/<run>/model_<iteration>" \
  --env.task Go1-DAWN-Play-v0 \
  --env.num-envs 4 \
  --video.enable
```

Playback saves a video in `logs/DAWN/<run>/videos/play/` and exits when recording is complete.
