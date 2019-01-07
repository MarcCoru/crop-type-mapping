#!/usr/bin/env bash

hyperparametercsv="/data/remote/hyperparams_conv1d_v2/hyperparams.csv/hyperparams_conv1d.csv"
store="/data/remote/early_rnn/conv1d"
epochs=60
switch_epoch=30
lossmode="twophase_linear_loss"
batchsize=256

for entropyfactor in 0.001 0.01 0.1; do
    for earlinessfactor in 0.5 0.6 0.7 0.8 0.9 1.0; do
        echo $(date) earliness $earlinessfactor, entropy $entropyfactor
        experiment=a${earlinessfactor}e$entropyfactor
        python train.py \
            --model Conv1D \
            --hyperparametercsv $hyperparametercsv \
            --loss_mode $lossmode \
            --experiment $experiment \
            --train_on trainvalid \
            --test_on test \
            --epochs $epochs \
            --switch_epoch $switch_epoch \
            --batchsize $batchsize \
            --overwrite \
            --workers 4 \
            --entropy-factor $entropyfactor \
            --earliness_factor $earlinessfactor \
            --store $store/$experiment

    done
done