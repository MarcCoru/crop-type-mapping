#!/usr/bin/env bash
epochs=100

for experiment in TUM_ALL_rnn TUM_GEN_rnn TUM_ALL_rnn_allclasses; do
python train.py -x $experiment --train_on trainvalid --test_on eval -e $epochs -b 1024 --dropout 0.25 -w 16 -i 0 --store /home/marc/projects/gafreport/images/data/ --test_every_n_epochs 5 --checkpoint_every_n_epochs 5
done

epochs=200

for experiment in TUM_ALL_transformer TUM_ALL_transformer_allclasses; do
python train.py -x $experiment --train_on trainvalid --test_on eval -e $epochs -b 256 --dropout 0.25 -w 16 -i 0 --store /home/marc/projects/gafreport/images/data/ --test_every_n_epochs 5 --checkpoint_every_n_epochs 5
done
