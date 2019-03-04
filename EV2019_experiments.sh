#!/usr/bin/env bash

cd src

hparams="-r 64 -n 4 -l 0.01 -e 100 -s 50 -b 1024 --dropout 0.5 -w 16 -i 5"

#python train.py -d BavarianCrops -m DualOutputRNN --loss_mode early_reward -x earlyreward --train_on train --test_on eval $hparams --dropout 0.5 -w 16 -i 0 -a .5 --store /data/EV2019 --overwrite --classmapping /home/marc/data/BavarianCrops/classmapping.csv.holl
#python train.py -d BavarianCrops -m DualOutputRNN --loss_mode twophase_cross_entropy -x twophasecrossentropy --train_on train --test_on eval $hparams --dropout 0.5 -w 16 -i 0 -a .5 --store /data/EV2019 --overwrite --classmapping /home/marc/data/BavarianCrops/classmapping.csv.holl

for alpha in 0 0.2 0.4 0.6 0.8 1;
do
for run in 0 1 2;
do
    echo
    #python train.py -d BavarianCrops -m DualOutputRNN --seed $run --loss_mode early_reward -x earlyreward-alpha${alpha}-run${run} --train_on train --test_on eval $hparams -a $alpha --store /data/EV2019 --overwrite --classmapping /home/marc/data/BavarianCrops/classmapping.csv.holl
    #python train.py -d BavarianCrops -m DualOutputRNN --seed $run --loss_mode twophase_cross_entropy -x twophasecrossentropy-alpha${alpha}-run${run} --train_on train --test_on eval $hparams -i 5 -a $alpha --store /data/EV2019 --overwrite --classmapping /home/marc/data/BavarianCrops/classmapping.csv.holl

done
done

for run in 0 1 2;
do
for alpha in 0 0.2 0.4 0.6 0.8 1;
do
for epsilon in 0 1 10;
do
    python train.py -d BavarianCrops -m DualOutputRNN --epsilon $epsilon --seed $run --loss_mode early_reward -x earlyreward-alpha${alpha}-epsilon${epsilon}-run${run} --train_on train --test_on eval $hparams -a $alpha --store /data/EV2019 --overwrite --classmapping /home/marc/data/BavarianCrops/classmapping.csv.holl
    python train.py -d BavarianCrops -m DualOutputRNN --epsilon $epsilon --seed $run --loss_mode twophase_cross_entropy -x twophasecrossentropy-alpha${alpha}-epsilon${epsilon}-run${run} --train_on train --test_on eval $hparams -i 5 -a $alpha --store /data/EV2019 --overwrite --classmapping /home/marc/data/BavarianCrops/classmapping.csv.holl
done
done
done