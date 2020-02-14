#!/bin/bash

WORKSPACE=${WORKSPACE:-"/workspace/"}

MODEL_NAME=${MODEL_NAME:-"LibriSpeech"}
MODEL_PATH=${MODEL_PATH:-"${WORKSPACE}/models/LibriSpeech/"}

DATASET=${DATASET:-"${WORKSPACE}/datasets/LibriSpeech/test_clean/"}

GPU=${GPU:-0}
COPY_THREADS=${COPY_THREADS:-0}

BEAM=${BEAM:-10}
LATTICE_BEAM=${LATTICE_BEAM:-7}
MAX_ACTIVE=${MAX_ACTIVE:-10000}
MAIN_Q_CAPACITY=${MAIN_Q_CAPACITY:-30000}
AUX_Q_CAPACITY=${AUX_Q_CAPACITY:-400000}

#query GPU memory
gpu_memory=`nvidia-smi -q -i $GPU | grep -A1 "FB Memory" | grep Total | tr -s " " | cut -d " " -f 4`
if [ $gpu_memory -ge 16000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-600}
elif [ $gpu_memory -ge 8000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-200}
elif [ $gpu_memory -ge 4000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-75}
else
  echo "ERROR not enough GPU memory to run benchmark."
  exit 1;
fi


