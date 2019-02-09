End-to-end Learning for Early Classification of Time Series (ELECTS)
===

### Experiments

#### Produce Table 2

```bash
$python winloosetables.py

& mori & relclass & edsc & ects \\
\cmidrule(lr){1-1}\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}
$0.6$ & 7 / \textbf{38} & \textbf{31} / 14 & \textbf{34} / 8 & \textbf{40} / 5 \\
$0.7$ & 3 / \textbf{42} & \textbf{28} / 17 & \textbf{28} / 14 & \textbf{35} / 10 \\
$0.8$ & 5 / \textbf{40} & \textbf{23} / 22 & \textbf{30} / 12 & \textbf{34} / 11 \\
$0.9$ & 12 / \textbf{33} & 19 / \textbf{26} & \textbf{33} / 9 & \textbf{26} / 19 \\
```
#### Figure 7 and Figure 8

writes data to `viz/csv` and some preliminary plots to `viz/png`
```angular2
python viz.py
```

#### Figures 5 and 6

start visdom server by typing `visdom` and browser navigate to `http://localhost:8097`

start training
```
python train.py -d TwoPatterns \
    -m Conv1D \
    --hyperparametercsv data/hyperparameter_conv1d.csv \
    --loss_mode twophase_linear_loss \
    --experiment test \
    --train_on train \
    --test_on valid \
    --batchsize 64 \
    --dropout 0.5 \ 
    --workers 4 \
    -i 1 \
    -a 0.6 \
    --overwrite \
    --test_every_n_epochs 1 \
    --store /tmp/run
```



### Scripts

##### Train single dataset with visdom logging

```
train.py
```

##### Train and Evaluate

```angular2
python eval.py -m DualOutputRNN -e 60 -b 128 -w 4 -i 20 -d TwoPatterns --dropout 0.2 --run evalearliness --store /tmp --hparams /data/remote/hyperparams_mori_fixed/hyperparams.csv --switch_epoch 0 --load_weights /data/remote/early_rnn/cross_entropy_e30/models/TwoPatterns/run/model_29.pth -i 1 --loss_mode twophase_linear_loss --earliness_factor .8 --entropy_factor 10
```

```angular2
python evalUCR.py -m DualOutputRNN --switch_epoch 30 -e 60 -b 64 -w 4 -i 20 --dropout .2 --root /data/remote/early_rnn --experiment 2phase --loss_mode twophase_linear_loss --earliness_factor 0.9 --weight_folder /data/remote/early_rnn/cross_entropy_e30/models --hparams /data/remote/hyperparams_mori_fixed/hyperparams.csv
```

```angular2
python utils/rayresultsparser.py "/data/remote/early_rnn/sota_comparison"
```

```angular2
python train.py -d TwoPatterns -m Conv1D --hyperparametercsv /data/remote/hyperparams_conv1d_v2/hyperparams.csv/hyperparams_conv1d.csv --loss_mode twophase_cross_entropy -x test --train_on trainvalid --test_on test -e 10 -s 5 -b 32 --dropout 0.5 -w 4 -i 1 --entropy-factor 0.01 -a .7 --test_every_n_epochs 1 --store /tmp/run
python train.py -d Symbols -m DualOutputRNN --loss_mode twophase_linear_loss -x test --train_on trainvalid --test_on test -r 20 -n 5 -l 0.1 -e 20 -s 10 -b 128 --dropout 0.5 -w 4 -i 1 -a .9 --store /tmp/rnn
python train.py -d TwoPatterns -m Conv1D --hyperparametercsv /data/remote/hyperparams_conv1d_v2/hyperparams.csv/hyperparams_conv1d.csv --loss_mode twophase_linear_loss -x test --train_on train --test_on valid -r 20 -n 1 -l 0.01 -e 15 -s 5 -b 64 --shapelet_width_increment 30 --dropout 0.5 -w 4 -i 1 --ptsepsilon 0 --entropy-factor 0 -a 0.6 --overwrite --test_every_n_epochs 1 --store /tmp/run
python train.py -d TwoPatterns -m Conv1D --hyperparametercsv /data/remote/hyperparams_conv1d_v2/hyperparams.csv/hyperparams_conv1d.csv --loss_mode twophase_cross_entropy -x test --train_on trainvalid --test_on test -e 10 -s 5 -b 32 --dropout 0.5 -w 4 -i 1 --entropy-factor 0.01 -a .7 --test_every_n_epochs 1 --store /tmp/run
```

##### tune hyperparameters

```angular2
tune.py test_conv1d -d experiments/morietal2017/UCR_dataset_names.txt -b 128 -c 2 -g 0
```

```angular2
python evalUCRwithRay.py /data/remote/hyperparams_conv1d_v2/hyperparams.csv/hyperparams_conv1d.csv -x entropy_pts -b 16 -c 2 -g .25 --skip-processed -r /tmp
```

```angular2
python runresultsparser.py
```


### Dependencies

### Visdom

The current version requires a Visdom server
```
visdom # open browser localhost 8097
```

### tests

```
python -m unittest discover tests
```

### DGX set up

delete unnecessary preinstalled depenencies...
```bash
sudo rm -r /opt/conda
sudo rm -rf /usr/local/onnx
sudo rm -rf /opt/caffe2End-to-end Learning for Early Classification of Time Series
sudo apt-get install rsync screen
```

### P100 set up

```
wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
bash Anaconda3-2018.12-Linux-x86_64.sh -b
echo ". $HOME/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
conda create -n early_rnn python=3.6 pip
conda activate early_rnn
pip install --no-cache-dir Cython
pip install numpy psutil
pip install --upgrade pip
pip install torch scikit-learn pandas visdom ray matplotlib seaborn
pip install git+https://github.com/marccoru/tslearn.git
```

#### Docker details

##### pull and run compiled container
I have pushed a pre-compiled container. So that should be enough

```
docker run marccoru/early_rnn
```

##### build container

```
docker build -t early_rnn .
```

##### push container 
(requires login to DockerHub)

```
docker tag early_rnn marccoru/early_rnn
docker push marccoru/early_rnn
```


