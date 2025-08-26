#!/bin/bash

# Exit immediately if any command fails
set -e

python tools/train.py configs/second/memory_optimized_adaptive_voxel_second.py
