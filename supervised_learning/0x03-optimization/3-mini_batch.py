#!/usr/bin/env python3
"""Function that trains a loaded neural network
model using mini-batch gradient descent"""

import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """Function that trains a loaded neural network
    model using mini-batch gradient descent"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        train_op = tf.get_collection("train_op")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        for i in range(epochs + 1):
            acc_t, cost_t = sess.run([accuracy, loss], feed_dict={
                                     x: X_train, y: Y_train})
            acc_v, cost_v = sess.run([accuracy, loss],
                                     feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_t))
            print("\tTraining Accuracy: {}".format(acc_t))
            print("\tValidation Cost: {}".format(cost_v))
            print("\tValidation Accuracy: {}".format(acc_v))
            if i < epochs:
                Xshf, Yshf = shuffle_data(X_train, Y_train)
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
        return saver.save(sess, save_path)
