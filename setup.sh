#!/bin/bash

git submodule update --init --recursive

apt-get update
apt-get install -y \
    build-essential \
    cmake \
    llvm \
    clang \
    libclang-dev

mkdir -p build && ./build.sh

chmod +x ./run.sh