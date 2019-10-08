#!/usr/bin/env bash
epochs=150

#experiments=isprs_tum_transformer isprs_tum_msresnet isprs_tum_rnn isprs_tum_tempcnn isprs_gaf_transformer isprs_gaf_msresnet isprs_gaf_rnn isprs_gaf_tempcnn
experiments=$1

for seed in 0 1 2; do
for experiment in $experiments; do
python train.py -x $experiment -e $epochs -b 256 -w 2 -i 0 --store /data/isprs/preraw2/$seed/ --test_every_n_epochs 1 --no-visdom --seed $seed --hparamset $seed
done

done

