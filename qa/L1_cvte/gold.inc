#!/bin/bash

declare -A GOLD_CER
declare -A GOLD_PERF

GOLD_CER=( 
  ["cvte"]="8.5" 
)

GOLD_PERF=(
  ["T4x1_cvte"]="0"
  ["P40x1_cvte"]="0"
  ["P100x1_cvte"]="0"
  ["V100x1_cvte"]="0"
)
