#!/usr/bin/env bash

# the experiment to pretrain the all models only on cross entropy
# commit 1951095cc66b1fdd91d73c4709a8a0116c55754c

python eval_mori.py -m DualOutputRNN -e 30 -b 64 -w 4 -i 20 -d TwoPatterns --dropout .2 --hparams /data/remote/hyperparams_mori_fixed/hyperparams.csv