#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
activate_script="$repo_root/.venv/bin/activate"
sim_setup="$repo_root/IsaacLab/_isaac_sim/setup_conda_env.sh"

if [[ ! -f "$activate_script" ]]; then
    echo "Run uv sync before setting up Isaac Sim." >&2
    exit 1
fi

if [[ ! -f "$sim_setup" ]]; then
    echo "Link the Isaac Sim installation at IsaacLab/_isaac_sim first." >&2
    exit 1
fi

if ! grep -Fq 'IsaacLab/_isaac_sim/setup_conda_env.sh' "$activate_script"; then
    cat >> "$activate_script" <<'EOF'

# Isaac Sim environment
if [ -f "${VIRTUAL_ENV}/../IsaacLab/_isaac_sim/setup_conda_env.sh" ]; then
    source "${VIRTUAL_ENV}/../IsaacLab/_isaac_sim/setup_conda_env.sh"
fi
EOF
fi

echo "Isaac Sim setup is available through .venv/bin/activate."
