Recurrent Neural Networks for Early Classification of Time Series
===

### DGX set up

delete unnecessary preinstalled depenencies...
```bash
sudo rm -r /opt/conda
sudo rm -rf /usr/local/onnx
sudo rm -rf /opt/caffe2
```

#### Unittests

```bash
python tests/run_unittests.py
```

#### Data



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

