#!/usr/bin/env bash
epochs=150
batchsize=512

#experiments=isprs_gaf_transformer_holl isprs_gaf_msresnet_holl isprs_gaf_rnn_holl isprs_gaf_tempcnn_holl isprs_tum_transformer isprs_tum_msresnet isprs_tum_rnn isprs_tum_tempcnn
case $1 in
prev)
# train models only on preprocessed HOLL region
for seed in 0 1 2; do
  for experiment in isprs_gaf_rnn_holl # isprs_gaf_transformer_holl isprs_gaf_msresnet_holl isprs_gaf_tempcnn_holl;
  do
  python train.py -x $experiment -e $epochs -b $batchsize -w 2 -i 0 --store /data/isprs/holl/$seed/ --test_every_n_epochs 1 --no-visdom --seed $seed --hparamset $seed --classmapping /data/GAFdataset/classmapping.isprs.csv --hyperparameterfolder /data/isprs/ray
  done
done

# train models on all raw regions
for seed in 0 1 2; do
  for experiment in isprs_tum_rnn_all # isprs_tum_transformer_all isprs_tum_msresnet_all isprs_tum_tempcnn_all;
  do
  python train.py -x $experiment -e $epochs -b $batchsize -w 2 -i 0 --store /data/isprs/holl/$seed/ --test_every_n_epochs 1 --no-visdom --seed $seed --hparamset $seed --classmapping /data/GAFdataset/classmapping.isprs.csv --hyperparameterfolder /data/isprs/ray
  done
done

# train models on all raw regions
for seed in 0 1 2; do
  for experiment in isprs_tum_rnn_holl isprs_tum_transformer_holl isprs_tum_msresnet_holl isprs_tum_tempcnn_holl;
  do
      python train.py -x $experiment -e $epochs -b $batchsize -w 2 -i 0 --store /data/isprs/holl/$seed/ --test_every_n_epochs 1 --no-visdom --seed $seed --hparamset $seed --classmapping /data/GAFdataset/classmapping.isprs.csv --hyperparameterfolder /data/isprs/ray
  done
done
;;
7)
export CUDA_VISIBLE_DEVICES=7
# train models on all raw regions
for seed in 0 1 2; do
  for experiment in isprs_tum_rnn_holl isprs_tum_transformer_holl isprs_tum_msresnet_holl isprs_tum_tempcnn_holl;
  do
      python train.py -x $experiment -e $epochs -b $batchsize -w 2 -i 0 --store /data/isprs/holl2/$seed/ --test_every_n_epochs 1 --no-visdom --seed $seed --hparamset $seed --classmapping /data/GAFdataset/classmapping.isprs2.csv --hyperparameterfolder /data/isprs/ray2
  done
done
;;
8)
export CUDA_VISIBLE_DEVICES=8
# train models only on preprocessed HOLL region
for seed in 0 1 2; do
  for experiment in isprs_gaf_rnn_holl isprs_gaf_transformer_holl isprs_gaf_msresnet_holl isprs_gaf_tempcnn_holl;
  do
    python train.py -x $experiment -e $epochs -b $batchsize -w 2 -i 0 --store /data/isprs/holl2/$seed/ --test_every_n_epochs 1 --no-visdom --seed $seed --hparamset $seed --classmapping /data/GAFdataset/classmapping.isprs2.csv --hyperparameterfolder /data/isprs/ray2
  done
done

# train models on all raw regions
for seed in 0 1 2; do
  for experiment in isprs_tum_rnn_all # isprs_tum_transformer_all isprs_tum_msresnet_all isprs_tum_tempcnn_all;
  do
  python train.py -x $experiment -e $epochs -b $batchsize -w 2 -i 0 --store /data/isprs/holl2/$seed/ --test_every_n_epochs 1 --no-visdom --seed $seed --hparamset $seed --classmapping /data/GAFdataset/classmapping.isprs2.csv --hyperparameterfolder /data/isprs/ray2
  done
done
;;
esac

