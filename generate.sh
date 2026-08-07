#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${repo_dir}"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
# shellcheck disable=SC1091
source "${repo_dir}/scripts/require_active_env.sh"
h3_require_active_env
exec "${repo_dir}/scripts/generate.sh" "$@"
