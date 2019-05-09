"""
Chapter 10 : Introduction to Artifical Neural Networks
-- The Perceptron
-- Multi-Layer Perceptron and Backpropagation


"""


# Training a DNN Using Plain TensorFlow

import tensorflow as tf
import numpy as np

# input, output, set the number of hidden neurons in each layer
n_inputs = 28*28 # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')


# create a neuron layer
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name='weights')
        b = tf.Variable(tf.zeros([n_neurons]), name='biases')
        z = tf.matmul(X, W) + b
        if activation == 'relu':
            return tf.nn.relu(z)
        else:
            return z

# use neuron_layer function to create a deep neuron network
# with tf.name_scope('dnn'):
#     hidden1 = neuron_layer(X, n_hidden1, 'hidden1', activation='relu')
# hidden2 = neuron_layer(hidden1, n_hidden2, 'hidden2', activation='relu')
# logits = neuron_layer(hidden2, n_outputs, 'outputs')


# use tf.layers.dense function to create a deep neuron network
with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X, n_hidden1, name='hidden1',
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2',
                              activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name='outputs')

# define the cost function
with tf.name_scope('loss'):
    xentroy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits)
    loss = tf.reduce_mean(xentroy, name='loss')

# define a GradientDescentOptimizer to minimize the cost function
learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# evaluate model, using accuracy
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# creat a node to initialize all variables
# creat a Saver to save our train model

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Execution Phase
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('path')

n_epochs = 400
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
        acc_test = accuracy.eval(feed_dict={X:mnist.test.images,
                                            y:mnist.test.labels})
        print(epoch, 'Training accuracy:', acc_train, 'Test accuracy:', acc_test)

    save_path = saver.save(sess, 'path/my_model_final.ckpt')

# USing the Neural Network
with tf.Session() as sess:
    saver.restore(sess, 'path/my_model_final.ckpt')
    X_new_scaled = [] # some new images (scaled from 0 to 1)
    Z = logits.eval(feed_dict={X:X_new_scaled})
    y_pred = np.argmax(Z, axis=1)


"""

 Complete Code
 
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

mnist = input_data.read_data_sets('path')

X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype('int')
y_test = mnist.test.labels.astype('int')

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                              name='hidden1')
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2',
                              activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name='outputs')

with tf.name_scope('loss'):
    xentroy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                             logits=logits)
    loss = tf.reduce_mean(xentroy, name='loss')

learning_rate = 0.01
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1) # 是否与真值一致，返回布尔值
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) # tf.cast将数据转化为0,1

init = tf.global_variables_initializer()

n_epochs = 20
batch_size = 50
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X:X_batch,
                                             y:y_batch})

        acc_train = accuracy.eavl(feed_dict={X:X_batch,
                                             y:y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                            y:mnist.test.labels})
        print(epoch, 'training accuracy:', acc_train, 'Test ccuracy:', acc_test)

