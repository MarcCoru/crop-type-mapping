for experiment in tempcnn_gaf msresnet_gaf rnn_gaf tempcnn_tum msresnet_tum; do
python tune.py -x $experiment -g 0.25 -c 2 -w 2 -b 64 --local_dir /data/isprs/ray_new/ -m 32 --classmapping /data/BavarianCrops/classmapping.isprs.csv
python tune.py -x $experiment -g 0.25 -c 2 -w 2 -b 64 --local_dir /data/isprs/ray_new2/ -m 32 --classmapping /data/BavarianCrops/classmapping.isprs2.csv
done
