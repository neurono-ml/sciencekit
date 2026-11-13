#!/usr/bin/env bash
# Publish every sciencekit crate in dependency order.
#
# A crate/version already present on crates.io is skipped, so the script is
# idempotent and safe to re-run after a partial failure (crates.io indexes
# asynchronously, and a dependent can only be published once its dependency is
# visible).
#
# Usage:
#   scripts/publish-crates.sh            # publish missing versions
#   scripts/publish-crates.sh --dry-run  # cargo publish --dry-run for each
#
# Requires CARGO_REGISTRY_TOKEN in the environment (CI) or in
# ~/.cargo/credentials.toml (local).
set -euo pipefail

CRATES=(
  sciencekit_common
  sciencekit_math
  sciencekit_metrics
  sciencekit_preprocessing
  sciencekit_impute
  sciencekit_linear_model
  sciencekit_neighbors
  sciencekit_cluster
  sciencekit_tree
  sciencekit_gpu
  sciencekit_gpu_opencl
  sciencekit_svm
  sciencekit_model_selection
  sciencekit_interop
  sciencekit_decomposition
  sciencekit_ensemble
  sciencekit_outlier
  sciencekit_telemetry
  sciencekit_pipeline
  sciencekit_ffi
  sciencekit_onnx
  sciencekit
)

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN="--dry-run"
fi

version_of() {
  cargo metadata --no-deps --format-version 1 \
    | python3 -c '
import json, sys
name = sys.argv[1]
metadata = json.load(sys.stdin)
for package in metadata["packages"]:
    if package["name"] == name:
        print(package["version"])
        break
' "$1"
}

published() {
  local name="$1" version="$2"
  local response
  response=$(curl -sS -A "sciencekit-release (https://github.com/neurono-ml/sciencekit)" \
    "https://crates.io/api/v1/crates/${name}/${version}")
  grep -q '"version"' <<<"${response}"
}

for crate in "${CRATES[@]}"; do
  version=$(version_of "${crate}")
  if published "${crate}" "${version}"; then
    echo "skip  ${crate} ${version} (already on crates.io)"
    continue
  fi
  echo "publish ${crate} ${version}"
  # crates.io rate-limits bursts of new crates with HTTP 429. Retry with
  # increasing backoff instead of failing the whole release; every attempt is
  # safe because a version can only be published once.
  delay=120
  attempt=1
  max_attempts=6
  while true; do
    output=""
    if output=$(cargo publish -p "${crate}" ${DRY_RUN} 2>&1); then
      echo "${output}"
      break
    fi
    if [[ "${DRY_RUN}" == "" ]] && grep -qE "status 429|Too Many Requests" <<<"${output}"; then
      if (( attempt >= max_attempts )); then
        echo "${output}"
        echo "crates.io kept rate-limiting ${crate} after ${attempt} attempts" >&2
        exit 1
      fi
      echo "rate limited on ${crate}; retrying in ${delay}s (attempt ${attempt}/${max_attempts})"
      sleep "${delay}"
      delay=$(( delay * 2 > 900 ? 900 : delay * 2 ))
      attempt=$(( attempt + 1 ))
      continue
    fi
    echo "${output}"
    exit 1
  done
  # crates.io needs a moment before the new version resolves for dependents.
  if [[ -z "${DRY_RUN}" ]]; then
    sleep 20
  fi
done

echo "all crates published (or already present)"
