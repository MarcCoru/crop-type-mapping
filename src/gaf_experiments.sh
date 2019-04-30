#!/usr/bin/env bash
epochs=60

for experiment in TUM_ALL_rnn TUM_HOLL_rnn TUM_ALL_transformer TUM_HOLL_transformer; do
python train.py -x $experiment --train_on trainvalid --test_on eval -n 1 -e $epochs -b 128 --dropout 0.25 -w 16 -i 1 --store /home/marc/projects/gafreport/images/data/ --checkpoint_every_n_epochs 5
done
