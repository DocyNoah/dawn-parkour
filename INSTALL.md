# Installation

## 1. Initialize Isaac Lab

From the repository root, initialize the pinned submodule:

```bash
git submodule update --init --recursive
```

## 2. Install Isaac Sim

Download the [Isaac Sim 5.1.0 Linux binary](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)
and extract it to `~/isaacsim`. Launch it once to complete its first-run setup:

```bash
~/isaacsim/isaac-sim.sh
```

Link the binary installation to Isaac Lab:

```bash
ln -s ~/isaacsim IsaacLab/_isaac_sim
```

If Isaac Sim is installed elsewhere, use that directory in the `ln -s` command.

## 3. Install the project

```bash
uv sync --extra cu128
bash setup_isaaclab_env.sh
source .venv/bin/activate
```

The setup script adds Isaac Sim's environment setup to `.venv/bin/activate`.
Source that activation script in each new shell before running DAWN.

## 4. Check Python imports

Check that both Isaac Sim and Isaac Lab are importable:

```bash
python -c "import isaacsim, isaaclab; print('Isaac Sim and Isaac Lab are available')"
```
