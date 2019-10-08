#!/usr/bin/env bash
epochs=150

#experiments=isprs_gaf_transformer_holl isprs_gaf_msresnet_holl isprs_gaf_rnn_holl isprs_gaf_tempcnn_holl isprs_tum_transformer isprs_tum_msresnet isprs_tum_rnn isprs_tum_tempcnn

# train models only on preprocessed HOLL region
for seed in 0 1 2; do
  for experiment in isprs_gaf_transformer_holl isprs_gaf_msresnet_holl isprs_gaf_rnn_holl isprs_gaf_tempcnn_holl; do
  python train.py -x $experiment -e $epochs -b 256 -w 2 -i 0 --store /data/isprs/holl/$seed/ --test_every_n_epochs 1 --no-visdom --seed $seed --hparamset $seed
  done
done

# train models on all raw regions
for seed in 0 1 2; do
  for experiment in isprs_tum_transformer isprs_tum_msresnet isprs_tum_rnn isprs_tum_tempcnn; do
  python train.py -x $experiment -e $epochs -b 256 -w 2 -i 0 --store /data/isprs/holl/$seed/ --test_every_n_epochs 1 --no-visdom --seed $seed --hparamset $seed
  done
done


