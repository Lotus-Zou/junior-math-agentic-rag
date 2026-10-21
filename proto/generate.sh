#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/math_agent.proto
