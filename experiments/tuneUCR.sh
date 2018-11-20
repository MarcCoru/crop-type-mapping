#!/usr/bin/env bash

for dataset in Adiac ArrowHead Beef BeetleFly BirdChicken Car CBF ChlorineConcentration CinCECGTorso Coffee Computers CricketX CricketY CricketZ DiatomSiz$
do
echo "$(date) current dataset: $dataset" >> $HOME/ray_results/datasets.log
python tune.py -d $dataset -b 64

done