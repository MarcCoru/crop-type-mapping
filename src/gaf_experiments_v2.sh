#!/usr/bin/env bash
epochs=150

#experiments=tumgaf_tum_transformer tumgaf_gaf_rnn_optical tumgaf_gaf_rnn_radar tumgaf_gaf_rnn_all tumgaf_tum_rnn tumgaf_gaf_rnn tumgaf_tum_tempcnn tumgaf_gaf_tempcnn tumgaf_tum_msresnet tumgaf_gaf_msresnet tumgaf_gaf_transformer
#tumgaf_tum_rnn tumgaf_tum_msresnet tumgaf_tum_tempcnn tumgaf_tum_transformer tumgaf_gaf_transformer_optical tumgaf_gaf_transformer_radar tumgaf_gaf_transformer_all

for experiment in tumgaf_gaf_transformer_optical tumgaf_gaf_rnn_optical tumgaf_gaf_msresnet_optical tumgaf_gaf_tempcnn_optical; do
python train.py -x $experiment -e $epochs -b 96 -w 2 -i 0 --store /data/gaf/runs --checkpoint_every_n_epochs 10 --test_every_n_epochs 1 --overwrite --no-visdom
done

