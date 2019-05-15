#!/usr/bin/env bash

/home/marc/miniconda3/envs/pytorch/bin/python utils/pred2shp.py /tmp/TUM_ALL_rnn_allclasses  /tmp/TUM_ALL_rnn_allclasses /home/marc/data/BavarianCrops/classmapping83.csv

/home/marc/miniconda3/envs/pytorch/bin/python utils/pred2shp.py /tmp/TUM_ALL_rnn  /tmp/TUM_ALL_rnn /home/marc/data/BavarianCrops/classmapping.csv.gaf
