#!/bin/bash

cd /opt/kaldi/tools/
make distclean
make -j 20

cd /opt/kaldi/src
make distclean
./configure --use-cuda=no --shared --mathlib=ATLAS --atlas-root=/usr/local 

make -j20 depend
make -j20


