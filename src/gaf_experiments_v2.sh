#!/usr/bin/env bash
epochs=150

for experiment in tumgaf_tum_transformer tumgaf_gaf_rnn_optical tumgaf_gaf_rnn_radar tumgaf_gaf_rnn_all tumgaf_tum_rnn tumgaf_gaf_rnn tumgaf_tum_tempcnn tumgaf_gaf_tempcnn tumgaf_tum_msresnet tumgaf_gaf_msresnet tumgaf_gaf_transformer; do
python train.py -x $experiment -e $epochs -b 96 -w 8 -i 0 --store /data/gaf/runs --checkpoint_every_n_epochs 10 --test_every_n_epochs 10
done

