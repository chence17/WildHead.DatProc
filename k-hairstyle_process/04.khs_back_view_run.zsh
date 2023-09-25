#!/bin/zsh

# Check if there are exactly two arguments provided
if [ $# -ne 4 ]; then
  echo "Usage: <arg1> <arg2> <arg3> <arg4>"
  exit 1
fi

CUDA_VISIBLE_DEVICES=$1 python k-hairstyle_process/04.khs_back_view.py -i /data_new/chence/K-Hairstyle/$2/rawset/meta_$3-$4.json
