#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f build/main ]]; then
  cmake -S . -B build
  cmake --build build --parallel
fi

exec ncu --target-processes all ./build/main "$@"
