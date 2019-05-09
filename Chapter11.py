
"""
Chapter11:  Training Deep Neural Networks

-- Vanishing/Exploding Gradients Problems
--

"""

# Vanishing/Exploding Gradients Problems

from functools import partial
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

path = 'E:/Julianna/PingAn/TensorFlow/image/'
mnist = input_data.read_data_sets(path)

batch_norm_momentum = 0.9
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=None, name='y')
training = tf.placeholder_with_default(False, shape=(), name='training')

with tf.name_scope('dnn'):
    he_init = tf.contrib.layer.variance_scaling_initializer()

    my_batch_norm_layer = partial(tf.layers.batch_normalization,
                                  training=training,
                                  momentum=batch_norm_momentum)

    my_dense_layer = partial(tf.layers.dense,
                             kernel_initializer=he_init)

    hidden1 = my_dense_layer(X, n_hidden1, name='hidden1')
    bn1 = tf.nn_elu(my_batch_norm_layer(hidden1))
    hidden2 = my_dense_layer(bn1, n_hidden2, name='hidden2')
    bn2 = tf.nn_elu(my_batch_norm_layer(hidden2))
    logists_before_bn = my_dense_layer(bn2, n_outputs, name='outputs')
    logists = my_batch_norm_layer(logists_before_bn)


with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logists)
    loss = tf.reduce_mean(xentropy, name='loss')

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    # new add
    # extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(extra_update_ops):
    #     training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logists, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epoches = 20
batch_size = 200

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoches):
        for iteration in range(mnist.train.num_examples//batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run([training_op, extra_update_ops],
                     feed_dict={training:True, X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})

        print(epoch, 'Test accuracy:', accuracy_val)



# Gradient Clipping

threshold = 1.0
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
              for grad, var in grads_and_vars]
training_op = optimizer.minimize(capped_gvs)


# Learning Rate Scheduling
n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=None, name='y')


with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name='hidden1')
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name='hidden2')
    logits = tf.layers.dense(hidden2, n_outputs, name='outputs')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

with tf.name_scope('train'):
    initial_learning_rate = 0.1
    decay_steps = 10000
    decay_rate = 1/10
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               decay_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epoches = 5
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoches):
        for iteration in range(mnist.train.num_examples//batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X:X_batch,
                                             y:y_batch})
        accuracy_val = accuracy.eval(feed_dict={X:mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, 'Test accuracy:', accuracy_val)
    save_path = saver.save(sess, 'path/my_model_final.ckpt')


# L1 & L2 Regularization

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=None, name='y')

scale = 0.001

my_dense_layer = partial(
    tf.layers.dense, activation=tf.nn.relu,
    kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))

with tf.name_scope('dnn'):
    hidden1 = my_dense_layer(X, n_hidden1, name='hidden1')
    hidden2 = my_dense_layer(hidden1, n_hidden2, name='hidden2')
    logits = my_dense_layer(hidden2, n_outputs, activations=None,
                            name='outputs')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits)
    base_loss = tf.reduce_mean(xentropy, name='avg_xentropy')
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_losses, name='loss')


with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

with tf.name_scope('train'):
    initial_learning_rate = 0.1
    decay_steps = 10000
    decay_rate = 1/10
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               decay_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epoches = 5
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoches):
        for iteration in range(mnist.train.num_examples//batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X:X_batch,
                                             y:y_batch})
        accuracy_val = accuracy.eval(feed_dict={X:mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, 'Test accuracy:', accuracy_val)
    save_path = saver.save(sess, 'path/my_model_final.ckpt')