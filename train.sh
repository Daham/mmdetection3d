#!/bin/bash

# Exit immediately if any command fails
set -e

python tools/train.py configs/second/learnable_adaptive_voxel_research.py
