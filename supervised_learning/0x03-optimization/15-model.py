#!/usr/bin/env python3
"""Function that builds, trains, and saves a neural network
model in tensorflow using Adam optimization, mini-batch
gradient descent, learning rate decay, and batch normalization"""

import tensorflow as tf
import numpy as np


def create_batch_norm_layer(prev, n, activation):
    """Function that creates a batch normalization
    layer for a neural network in tensorflow"""
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(units=n, name="layer", kernel_initializer=W)
    X = model(prev)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    mean, variance = tf.nn.moments(X, [0])
    norm = tf.nn.batch_normalization(X, mean, variance, offset=beta,
                                     scale=gamma, variance_epsilon=1e-8)
    return activation(norm)


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """that creates the training operation for a neural network
    in tensorflow using the Adam optimization algorithm"""
    optimizer = tf.train.AdamOptimizer(alpha, beta1=beta1, beta2=beta2,
                                       epsilon=epsilon)
    return (optimizer.minimize(loss))


def create_layer(prev, n, activation):
    """Function that returns the tensor output of the layer"""
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(units=n, activation=activation,
                            name="layer", kernel_initializer=W)
    return model(prev)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Function that creates a learning rate decay
operation in tensorflow using inverse time decay"""
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)


def forward_prop(x, layers=[], activations=[]):
    """Function that creates the forward
    propagation graph for the neural network"""
    A = create_batch_norm_layer(x, layers[0], activations[0])
    for i in range(1, len(layers)):
        if i != len(layers) - 1:
            A = create_batch_norm_layer(A, layers[i], activations[i])
        else:
            A = create_layer(A, layers[i], activations[i])
    return A


def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of a prediction"""
    y_m = tf.argmax(y, axis=1)
    yp_m = tf.argmax(y_pred, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(y_m, yp_m), 'float32'))


def shuffle_data(X, Y):
    """Function that shuffles the data
    points in two matrices the same way"""
    random = np.random.permutation(X.shape[0])
    return (X[random], Y[random])


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """Function that builds, trains, and saves a neural network
    model in tensorflow using Adam optimization, mini-batch
    gradient descent, learning rate decay, and batch normalization"""
    x = tf.placeholder(tf.float32, shape=(None, Data_train[0].shape[1]))
    y = tf.placeholder(tf.float32, shape=(None, Data_train[1].shape[1]))
    y_pred = forward_prop(x, layers, activations)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    global_step = tf.Variable(0, trainable=False)
    increment_global_step_op = tf.assign(global_step, global_step + 1)
    alpha_new = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train', train_op)
    tf.add_to_collection('alpha', alpha_new)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs + 1):
            acc_t, cost_t = sess.run([accuracy, loss], feed_dict={
                                     x: Data_train[0], y: Data_train[1]})
            acc_v, cost_v = sess.run([accuracy, loss], feed_dict={
                                     x: Data_valid[0], y: Data_valid[1]})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_t))
            print("\tTraining Accuracy: {}".format(acc_t))
            print("\tValidation Cost: {}".format(cost_v))
            print("\tValidation Accuracy: {}".format(acc_v))
            if i < epochs:
                Xshf, Yshf = shuffle_data(Data_train[0], Data_train[1])
                batch = Xshf.shape[0]
                start = 0
                step = 1
                while batch > 0:
                    if batch - batch_size < 0:
                        end = Xshf.shape[0]
                    else:
                        end = start + batch_size
                    X = Xshf[start:end]
                    Y = Yshf[start:end]
                    sess.run(train_op, feed_dict={x: X, y: Y})

                    if step % 100 == 0:
                        step_cost = sess.run(loss, feed_dict={x: X, y: Y})
                        step_acc = sess.run(accuracy, feed_dict={x: X, y: Y})
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_acc))
                    step = step + 1
                    batch = batch - batch_size
                    start = start + batch_size
            sess.run(increment_global_step_op)
            alpha = sess.run(alpha_new)
        return saver.save(sess, save_path)
