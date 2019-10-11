#!/usr/bin/env bash
epochs=150
batchsize=512
#experiments=isprs_gaf_transformer_krum isprs_gaf_msresnet_krum isprs_gaf_rnn_krum isprs_gaf_tempcnn_krum isprs_tum_transformer isprs_tum_msresnet isprs_tum_rnn isprs_tum_tempcnn

case $1 in
1)
  CUDA_VISIBLE_DEVICES=0
# train models only on preprocessed krum region
for seed in 0 1 2; do
  for experiment in isprs_gaf_transformer_krum isprs_gaf_msresnet_krum isprs_gaf_rnn_krum isprs_gaf_tempcnn_krum; do
  python train.py -x $experiment -e $epochs -b $batchsize -w 2 -i 0 --store /data/isprs/krum/$seed/ --test_every_n_epochs 1 --no-visdom --seed $seed --hparamset $seed --classmapping /data/GAFdataset/classmapping.isprs.csv  --hyperparameterfolder /data/isprs/ray
  done
done
;;
2)
CUDA_VISIBLE_DEVICES=1
# train models on allkrum raw regions
for seed in 0 1 2; do
  for experiment in isprs_tum_transformer_allkrum isprs_tum_msresnet_allkrum isprs_tum_rnn_allkrum isprs_tum_tempcnn_allkrum; do
  python train.py -x $experiment -e $epochs -b $batchsize -w 2 -i 0 --store /data/isprs/krum/$seed/ --test_every_n_epochs 1 --no-visdom --seed $seed --hparamset $seed --classmapping /data/GAFdataset/classmapping.isprs.csv --hyperparameterfolder /data/isprs/ray
  done
done
;;
3)
  CUDA_VISIBLE_DEVICES=2
# train models on allkrum raw regions
for seed in 0 1 2; do
  for experiment in isprs_tum_transformer_krum isprs_tum_msresnet_krum isprs_tum_rnn_krum isprs_tum_tempcnn_krum; do
      python train.py -x $experiment -e $epochs -b $batchsize -w 2 -i 0 --store /data/isprs/krum/$seed/ --test_every_n_epochs 1 --no-visdom --seed $seed --hparamset $seed --classmapping /data/GAFdataset/classmapping.isprs.csv --hyperparameterfolder /data/isprs/ray
  done
done
;;
4)
  CUDA_VISIBLE_DEVICES=3
# train models only on preprocessed krum region
for seed in 0 1 2; do
  for experiment in isprs_gaf_transformer_krum isprs_gaf_msresnet_krum isprs_gaf_rnn_krum isprs_gaf_tempcnn_krum; do
  python train.py -x $experiment -e $epochs -b $batchsize -w 2 -i 0 --store /data/isprs/krum2/$seed/ --test_every_n_epochs 1 --no-visdom --seed $seed --hparamset $seed --classmapping /data/GAFdataset/classmapping.isprs2.csv --hyperparameterfolder /data/isprs/ray2
  done
done
;;
5)
  CUDA_VISIBLE_DEVICES=4
# train models on allkrum raw regions
for seed in 0 1 2; do
  for experiment in isprs_tum_transformer_allkrum isprs_tum_msresnet_allkrum isprs_tum_rnn_allkrum isprs_tum_tempcnn_allkrum; do
  python train.py -x $experiment -e $epochs -b $batchsize -w 2 -i 0 --store /data/isprs/krum2/$seed/ --test_every_n_epochs 1 --no-visdom --seed $seed --hparamset $seed --classmapping /data/GAFdataset/classmapping.isprs2.csv --hyperparameterfolder /data/isprs/ray2
  done
done
;;
6)
  CUDA_VISIBLE_DEVICES=5
# train models on allkrum raw regions
for seed in 0 1 2; do
  for experiment in isprs_tum_transformer_krum isprs_tum_msresnet_krum isprs_tum_rnn_krum isprs_tum_tempcnn_krum; do
      python train.py -x $experiment -e $epochs -b $batchsize -w 2 -i 0 --store /data/isprs/krum2/$seed/ --test_every_n_epochs 1 --no-visdom --seed $seed --hparamset $seed --classmapping /data/GAFdataset/classmapping.isprs2.csv --hyperparameterfolder /data/isprs/ray2
  done
done
esac
