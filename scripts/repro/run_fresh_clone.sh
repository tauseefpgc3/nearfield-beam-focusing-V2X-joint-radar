#!/usr/bin/env bash
# D-034 C1: end-to-end regeneration of one simulated test-set table in a fresh clone.
# usage: bash scripts/repro/run_fresh_clone.sh <checkpoint dir (Zenodo package: phase3/)>
# Steps: users + channels (p1_channels), split (p1_lidar_split), LiDAR BEV (lidar_v2), then
# the table (c1_table.py). Needs a CUDA GPU for the LiDAR renderer and the networks.
set -e; cd "$(dirname "$0")/../.."; export PYTHONIOENCODING=utf-8
python scripts/phase2/p1_channels.py
python scripts/phase2/p1_lidar_split.py
python scripts/phase3/lidar_v2.py
python scripts/repro/c1_table.py --ckpt "$1"
