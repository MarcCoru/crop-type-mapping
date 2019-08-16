#!/usr/bin/env bash
epochs=300

for experiment in tumgaf_gaf_rnn_optical tumgaf_gaf_rnn_radar tumgaf_gaf_rnn_all tumgaf_tum_rnn tumgaf_gaf_rnn tumgaf_tum_msresnet tumgaf_gaf_msresnet tumgaf_tum_transformer tumgaf_gaf_transformer; do
python train.py -x $experiment --train_on trainvalid --test_on eval -e $epochs -b 128 --dropout 0.25 -w 1 -i 0 --store /data/tumgaf/ --checkpoint_every_n_epochs 10 --test_every_n_epochs 1 --samplet 50 --overwrite
done

