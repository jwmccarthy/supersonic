#!/usr/bin/env bash
set -euo pipefail

rebuild=0
clean=0
sims=1024
numB=4
numO=4
seed=123
iter=10000

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b)
      rebuild=1
      shift
      ;;
    -c)
      clean=1
      shift
      ;;
    --sims)
      sims=${2:?"Missing value for $1"}
      shift 2
      ;;
    --numB)
      numB=${2:?"Missing value for $1"}
      shift 2
      ;;
    --numO)
      numO=${2:?"Missing value for $1"}
      shift 2
      ;;
    --seed)
      seed=${2:?"Missing value for $1"}
      shift 2
      ;;
    --iter)
      iter=${2:?"Missing value for $1"}
      shift 2
      ;;
    *)
      exit 2
      ;;
  esac
done

if [[ ${clean} -eq 1 ]]; then
  rm -rf build
fi

if [[ ${rebuild} -eq 1 || ! -f build/main ]]; then
  cmake -S . -B build
  cmake --build build --parallel
fi

./build/main "${sims}" "${numB}" "${numO}" "${seed}" "${iter}"
