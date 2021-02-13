# 0x03. Optimization

### Description

This project is about doing some optimizations to the deep learning algorithms.

### General Objectives

* What is a hyperparameter?
* How and why do you normalize your input data?
* What is a saddle point?
* What is stochastic gradient descent?
* What is mini-batch gradient descent?
* What is a moving average? How do you implement it?
* What is gradient descent with momentum? How do you implement it?
* What is RMSProp? How do you implement it?
* What is Adam optimization? How do you implement it?
* What is learning rate decay? How do you implement it?
* What is batch normalization? How do you implement it?

### Mandatory Tasks

| File | Description |
| ------ | ------ |
| [0-norm_constants.py](0-norm_constants.py) | Calculates the normalization (standardization) constants of a matrix. |
| [1-normalize.py](1-normalize.py) | Normalizes (standardizes) a matrix. |
| [2-shuffle_data.py](2-shuffle_data.py) | Shuffles the data points in two matrices the same way. |
| [3-mini_batch.py](3-mini_batch.py) | Trains a loaded neural network model using mini-batch gradient descent. |
| [4-moving_average.py](4-moving_average.py) | Calculates the weighted moving average of a data set. |
| [5-momentum.py](5-momentum.py) | Updates a variable using the gradient descent with momentum optimization algorithm. |
| [6-momentum.py](6-momentum.py) | Creates the training operation for a neural network in tensorflow using the gradient descent with momentum optimization algorithm. |
| [7-RMSProp.py](7-RMSProp.py) | Updates a variable using the RMSProp optimization algorithm. |
| [8-RMSProp.py](8-RMSProp.py) | Creates the training operation for a neural network in tensorflow using the RMSProp optimization algorithm. |
| [9-Adam.py](9-Adam.py) | Updates a variable in place using the Adam optimization algorithm. |
| [10-Adam.py](10-Adam.py) | Creates the training operation for a neural network in tensorflow using the Adam optimization algorithm. |
| [11-learning_rate_decay.py](11-learning_rate_decay.py) | Updates the learning rate using inverse time decay in numpy. |
| [12-learning_rate_decay.py](12-learning_rate_decay.py) | Creates a learning rate decay operation in tensorflow using inverse time decay. |
| [13-batch_norm.py](13-batch_norm.py) | Normalizes an unactivated output of a neural network using batch normalization. |
| [14-batch_norm.py](14-batch_norm.py) | Creates a batch normalization layer for a neural network in tensorflow. |
| [15-model.py](15-model.py) | Builds, trains, and saves a neural network model in tensorflow using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization. |
