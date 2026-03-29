#!/bin/bash
# --- Rusty Machine WSL2 Compilation Wrapper ---
# WSL abstracts GPU driver pointers into /usr/lib/wsl/lib instead of /usr/local/cuda.
# Rust's low-level CUDA crates dynamically panic on build if they can't establish continuous lib64 paths.
# We map a direct localized symlink fake-root to securely traverse these dependencies.

set -e

echo ">>> Initializing WSL2 Target Topography..."
mkdir -p .wsl_cuda/lib64
ln -sf /usr/lib/wsl/lib/*.so* .wsl_cuda/lib64/ 2>/dev/null
ln -sf /usr/lib/x86_64-linux-gnu/libcu*.so* .wsl_cuda/lib64/ 2>/dev/null

export CUDA_PATH=/usr
export CUDA_LIBRARY_PATH="$(pwd)/.wsl_cuda"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"

echo ">>> Building Native Extension..."
uv sync
uv pip install -e .

echo ">>> Pre-Flight Checks Passed. Launching Aura."
uv run streamlit run app.py
