# Cifar-10 training & test tutorial

Created by: Won Seok, Choi

Commit Date: '17. 8. 24.

Version: Python3, Tensorflow


## Network Configuration
Input: cifar-10 image: 32x32x3

Network: 3 Convolution layers, 3 Pooling layers, 3 Fully-connected layers

Optimizer: AdamOptimizer


## Set Hyper-Parameters & Usage

There are 3 hyper-parameters to choose from. `Epochs, Batch size`, and `Drop out probability`.

For example, set `Epochs: 10, Batch size: 128, Drop out: 0.75`
```shell
python cifar-10.py -e 10 -b 128 -d 0.75
```

If you do not set anything, the default settings will be used.
```shell
python cifar-10.py
```
default: `Epochs: 10, Batch size: 128, Drop out: 0.75`
