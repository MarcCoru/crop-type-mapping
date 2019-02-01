#!/usr/bin/env bash

rsync -avz -e "ssh" marc@uni:/data/remote/hyperparams_conv1d_v2/hyperparams.csv/hyperparams_conv1d.csv ./hyperparams_conv1d_v2.csv