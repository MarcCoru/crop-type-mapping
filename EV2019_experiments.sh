#!/usr/bin/env bash

cd src

hparams="-r 64 -n 4 -l 0.01 -e 100 -s 50 -b 1024"

#python train.py -d BavarianCrops -m DualOutputRNN --loss_mode early_reward -x earlyreward --train_on train --test_on eval $hparams --dropout 0.5 -w 16 -i 0 -a .5 --store /data/EV2019 --overwrite --classmapping /home/marc/data/BavarianCrops/classmapping.csv.holl
#python train.py -d BavarianCrops -m DualOutputRNN --loss_mode twophase_cross_entropy -x twophasecrossentropy --train_on train --test_on eval $hparams --dropout 0.5 -w 16 -i 0 -a .5 --store /data/EV2019 --overwrite --classmapping /home/marc/data/BavarianCrops/classmapping.csv.holl

for alpha in 0 0.2 0.4 0.6 0.8 1;
do
#python train.py -d BavarianCrops -m DualOutputRNN --loss_mode early_reward -x earlyrewarda$alpha --train_on train --test_on eval $hparams --dropout 0.5 -w 16 -i 5 -a $alpha --store /data/EV2019 --overwrite --classmapping /home/marc/data/BavarianCrops/classmapping.csv.holl
python train.py -d BavarianCrops -m DualOutputRNN --loss_mode twophase_cross_entropy -x twophasecrossentropy$alpha --train_on train --test_on eval $hparams --dropout 0.5 -w 16 -i 5 -a $alpha --store /data/EV2019 --overwrite --classmapping /home/marc/data/BavarianCrops/classmapping.csv.holl
done