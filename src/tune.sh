for experiment in transformer_gaf tempcnn_gaf msresnet_gaf rnn_gaf transformer_tum tempcnn_tum msresnet_tum rnn_tum; do
python tune.py -x $experiment -g 0.25 -c 2 -w 2 -b 64 --local_dir /data/isprs/ray/ -m 32
done
