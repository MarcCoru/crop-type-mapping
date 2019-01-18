#!/usr/bin/env bash

fixedargs="--hyperparametercsv /data/remote/hyperparams_conv1d_v2/hyperparams.csv/hyperparams_conv1d.csv
    -d TwoPatterns -m Conv1D --loss_mode twophase_linear_loss --train_on trainvalid --test_on test
    -l 0.01 -e 15 -s 5 --dropout 0.5 -b 64 -w 4 -i 2 -a 0.6 --overwrite --test_every_n_epochs 1 --store /tmp/run"

echo
echo args with regularization
echo
echo $fixedargs -x regularization_with --entropy-factor 0.01 --epsilon 5
echo
echo args without regularization
echo
echo $fixedargs -x regularization_without --entropy-factor 0.00 --epsilon 0
echo
