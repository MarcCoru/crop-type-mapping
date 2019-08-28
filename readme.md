Crop Type Mapping
===

A respitory for classification of crop types using Satellite Time Series

Currently four models are implemented

* Recurrent Neural Net (LSTM) `src/models/rnn.py`
* Transformer `src/models/transformerEncoder.py
* TempCNN (Pelleter et al., 2019) `src/models/tempcnn.py`
* Multi-scale ResNet `src/models/msresnet.py`

When exploring this repository, start with `src/train.py`.
This loads the `dataloader`