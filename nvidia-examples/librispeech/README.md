LibriSpeech Example
===========

## Input Requirements

Expected Format:  wav
Expected Bitrate:  16000

## How to run

This example is an english model trained on the LibriSpeech dataset.  


To run this example you will first have to prepare the model

```
cd /workspace/nvidia-examples/librispeech/
./prepare.sh 
```

Once the model is prepared you can run a speech to text benchmark as follows:

```
cd /workspace/nvidia-examples/librispeech/
./run_benchmark.sh
```

You can use the ONLINE=1 option to run the online bimary:

```
ONLINE=1 ./run_benchmark.sh
```
