#!/usr/bin/env bash

hyperparametercsv="/data/remote/hyperparams_conv1d_v2/hyperparams.csv/hyperparams_conv1d.csv"
store="/data/remote/early_rnn/conv1d_fixed"
epochs=60
switch_epoch=30
batchsize=128

for entropyfactor in 0.01 0.1 0.001; do
    for earlinessfactor in 0.6 0.7 0.8 0.9; do
        for lossmode in "twophase_cross_entropy" "twophase_linear_loss"; do
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
                --test_every_n_epochs $switch_epoch \
                --batchsize $batchsize \
                --overwrite \
                --dropout 0.5 \
                --workers 2 \
                --entropy-factor $entropyfactor \
                --earliness_factor $earlinessfactor \
                --store $store/$lossmode/$experiment \
                --skip
        done
    done
done