End-to-end Learning for Early Classification of Time Series
===

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

#### Unittests

```bash
cd tests
python -m unittest discover
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

