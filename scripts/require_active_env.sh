#!/usr/bin/env bash

h3_require_active_env() {
  [[ "${H3_PLATFORM:-}" == "nibi" ]] || return 0

  local expected_env=${H3_ENV_DIR:-}
  local active_env=${VIRTUAL_ENV:-}
  if [[ -z "${expected_env}" || -z "${active_env}" ]]; then
    echo "Activate the MiniMax H3 environment first:" >&2
    echo "  source .venv/bin/activate" >&2
    return 1
  fi

  local expected_real active_real
  expected_real=$(python3 - "${expected_env}" <<'PY'
import os
import sys

print(os.path.realpath(sys.argv[1]))
PY
)
  active_real=$(python3 - "${active_env}" <<'PY'
import os
import sys

print(os.path.realpath(sys.argv[1]))
PY
)
  if [[ "${active_real}" != "${expected_real}" ]]; then
    echo "A different Python environment is active: ${active_env}" >&2
    echo "Activate the MiniMax H3 environment instead:" >&2
    echo "  source .venv/bin/activate" >&2
    return 1
  fi
}
