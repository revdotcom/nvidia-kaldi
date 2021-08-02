#!/bin/bash

cd /opt/kaldi/src
make -j$(nproc) test

