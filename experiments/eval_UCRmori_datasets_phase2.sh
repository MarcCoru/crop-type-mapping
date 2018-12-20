#!/usr/bin/env bash

# requires hyperparameter csv file
# retraines on the full UCR training dataset and evaluates on the evaluation dataset
# loads pretrained weights from eval_mori_cross_entropy_only.sh experiment (epoch 30)
# and trains on accuracy + earliness with different earliness factors

for earliness_factor in 0.5 0.6 0.7 0.8 0.9 1.0; do
python eval_mori.py\
    -m DualOutputRNN \
    --switch_epoch 30 \
    -e 60 \
    -b 64 \
    -w 4 \
    -i 20 \
    --dropout .2 \
    --root /data/remote/early_rnn --experiment phase2_a$earliness_factor --loss_mode twophase_linear_loss \
    --earliness_factor $earliness_factor \
    --weight_folder /data/remote/early_rnn/cross_entropy_e30/models \
    --hparams /data/remote/hyperparams_mori_fixed/hyperparams.csv
done