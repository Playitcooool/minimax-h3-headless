#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${repo_dir}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "MiniMax H3 inference setup requires a Linux GPU server." >&2
  exit 1
fi

missing_system=()
for command_name in curl git ffmpeg python3; do
  command -v "${command_name}" >/dev/null 2>&1 || missing_system+=("${command_name}")
done

if ((${#missing_system[@]} > 0)); then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Install these system packages, then rerun setup.sh: ${missing_system[*]}" >&2
    exit 1
  fi
  if ((EUID == 0)); then
    apt_prefix=()
  elif command -v sudo >/dev/null 2>&1; then
    apt_prefix=(sudo)
  else
    echo "sudo is required to install: ${missing_system[*]}" >&2
    exit 1
  fi
  "${apt_prefix[@]}" apt-get update
  "${apt_prefix[@]}" apt-get install -y "${missing_system[@]}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || { echo "uv installation failed." >&2; exit 1; }

echo "Setting up the gateway ..."
scripts/bootstrap_gateway.sh

echo "Setting up SGLang ..."
scripts/bootstrap_sglang.sh

mkdir -p outputs logs .run models

echo
echo "Setup complete. Next run: ./download_models.sh"
