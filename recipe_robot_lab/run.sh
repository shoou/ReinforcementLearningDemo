#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_script="$task_root/recipe_robot_lab/run_case.py"

if command -v python >/dev/null 2>&1 &&
   python -c 'import numpy, scipy, torch, pybullet, pybullet_data, PIL' >/dev/null 2>&1; then
    exec python "$task_script" "$@"
fi

if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)"
    case_python="$conda_base/bin/python3.13"
    physics_packages="$task_root/robot_arm_rl/.venv-physics/lib/python3.13/site-packages"
    if [[ -x "$case_python" ]]; then
        export PYTHONPATH="$physics_packages${PYTHONPATH:+:$PYTHONPATH}"
        if "$case_python" -c 'import numpy, scipy, torch, pybullet, pybullet_data, PIL' >/dev/null 2>&1; then
            exec "$case_python" "$task_script" "$@"
        fi
    fi
fi

printf '%s\n' '未找到同时具有 NumPy、SciPy、PyTorch、PyBullet 和 Pillow 的 Python 环境。' >&2
printf '%s\n' '请按 recipe_robot_lab/README.md 的依赖说明安装后重试。' >&2
exit 1
