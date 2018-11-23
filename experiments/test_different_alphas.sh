#!/usr/bin/env bash

for alpha in 0.5 0.6 0.7 0.8 0.9 1.0; do
    python ../train.py -m DualOutputRNN --loss_mode twophase_linear_loss -r 120 -n 2 -l 0.0005 -e 10 -b 30 -w 4 -i 20 \
     -d synthetic --use_batchnorm -a $alpha --store /data/early_rnn/alpha --run synthetic_a$alpha
done