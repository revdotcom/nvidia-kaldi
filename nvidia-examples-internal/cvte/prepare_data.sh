#!/bin/bash

source ./default_parameters.inc
source ../default_parameters.inc
model=cvte

data=${1:-$WORKSPACE/data/}
datasets=$WORKSPACE/datasets/
dataset=thchs30
models=$WORKSPACE/models/

# base url for downloads.
data_url=www.openslr.org/resources/18/data_thchs30.tgz
model_url=www.kaldi-asr.org/models/2/0002_cvte_chain_model_v2.tar.gz

mkdir -p $data
mkdir -p $models/$model
mkdir -p $datasets/$dataset

# you might not want to do this for interactive shells.
set -e

if [[ "$SKIP_DATA_DOWNLOAD" -ne "1" ]]; then
  echo ----------- Fetching data --------------

  pushd $data
  wget $data_url
  tarfile_name=$(echo $data_url | rev | cut -d/ -f1 | rev)
  echo $tarfile_name

  echo ----------- Extracting data files ------
  tar -xzf $tarfile_name --wildcards "data_thchs30/test/*"
  tar -xzf $tarfile_name --wildcards "data_thchs30/data/D*.trn"
  popd >&/dev/null

  echo ----------- Building dataset -----------
  #Kaldi provides a data prep script for thchs30
  #Builds wav.scp, utt2spk, spk2utt and text
  $KALDI_ROOT/egs/multi_cn/s5/local/thchs-30_data_prep.sh $data/data_thchs30 $datasets/$dataset | cat
  echo ----------- Making a smaller test set --
  #Building a smaller test set for shorter runtimes
  # To enable the full test set, set DATASET to $datasets/$dataset/test in
  # default_parameters.inc
  mkdir -p $datasets/$dataset/small
  cp $datasets/$dataset/test/* $datasets/$dataset/small/
  head -200 $datasets/$dataset/test/wav.scp > $datasets/$dataset/small/wav.scp
  head -200 $datasets/$dataset/test/utt2spk > $datasets/$dataset/small/utt2spk
  head -200 $datasets/$dataset/test/text > $datasets/$dataset/small/text

fi

if [[ "$SKIP_MODEL_DOWNLOAD" -ne "1" ]]; then
  echo ----------- Fetching pre-trained model -----------
  pushd $models >&/dev/null
  wget $model_url
  tgzfile_name=$(echo $model_url | rev | cut -d/ -f1 | rev )
  echo ----------- Extracting model -----------------
  gzip -d $tgzfile_name
  tarfile_name=0002_cvte_chain_model_v2.tar
  
  tar -xf $tarfile_name -C $models/$model cvte/s5/exp/chain/tdnn/graph \
         cvte/s5/exp/chain/tdnn/final.mdl cvte/s5/exp/chain/tdnn/global_cmvn \
         cvte/s5/exp/chain/tdnn/frame_subsampling_factor \
         cvte/s5/exp/chain/tdnn/cmvn_opts cvte/s5/conf/fbank.conf 
  mv cvte/cvte/s5/exp/chain/tdnn/graph/* $model/
  mv cvte/cvte/s5/exp/chain/tdnn/* $model/
  mkdir -p $models/$model/conf/
  mv cvte/cvte/s5/conf/fbank.conf $model/conf/fbank.conf
  cat $model/cmvn_opts | awk '{for (f=1;f<NF;f++) { print $f }}' > $model/conf/cmvn.conf
  echo ----------- Building conf files --------------
  echo "--feature-type=fbank" > $model/conf/online.conf
  echo "--fbank-config=$models/$model/conf/fbank.conf" >> $model/conf/online.conf
  echo "--cmvn-config=$models/$model/conf/cmvn.conf" >> $model/conf/online.conf
  echo "--global_cmvn_stats=$models/$model/global_cmvn" >> $model/conf/online.conf
  popd >&/dev/null
fi

ln -s ../run_benchmark.sh

