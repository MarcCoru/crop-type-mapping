Recurrent Neural Networks for Early Classification of Time Series
===

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
