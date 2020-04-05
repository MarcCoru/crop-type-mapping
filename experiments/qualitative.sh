#!/usr/bin/env bash

seed=0
epochs=150
hyperparameterfolder="/home/marc/remote/crop-type-mapping/models/tune/12classes"
classmapping="/home/marc/remote/crop-type-mapping/data/BavarianCrops/classmapping12.csv"
for experiment in "isprs_tum_tempcnn" "isprs_tum_transformer" "isprs_tum_msresnet" "isprs_tum_rnn"; do
python train.py -x $experiment -e $epochs -b 256 -w 0 -i 0 --checkpoint_every_n_epochs 5 --store /tmp/qualitative --test_every_n_epochs 1 --no-visdom --seed $seed --hparamset $seed --classmapping $classmapping --hyperparameterfolder $hyperparameterfolder
done