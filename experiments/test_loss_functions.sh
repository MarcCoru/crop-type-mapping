#!/usr/bin/env bash
for loss_mode in early_reward twophase_early_reward twophase_linear_loss twophase_early_simple;
do
python ../train.py --loss_mode $loss_mode -m DualOutputRNN -x testloss -r 120 -n 2 -l 0.001 -e 10 -b 30 -w 4 -d synthetic --use_batchnorm -a .75 -s 5
done