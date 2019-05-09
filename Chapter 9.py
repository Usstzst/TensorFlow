"""
Chapter 9: Up and Running with TensorFlow
-- Implementint Gradient Descent
-- Feeding Data to the Training Algorithm
-- Saving and Restoring Models
-- Visualizing the Graph and Training Curves Using TensorBoard
-- Name Scopes
-- Modularity
-- Sharing Variables

"""

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m,1)), scaled_housing_data]

from datetime import datetime

now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = '{}/run-{}/'.format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n+1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0,seed=42), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')



n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m/batch_size))


def fetch_batch(epoch, batch_index, batch_size):
    know = np.random.seed(epoch*n_batches+batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1,1)[indices]
    return X_batch, y_batch


with tf.name_scope('loss') as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')


optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X:X_batch, y:y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)

            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
    best_theta = theta.eval()

    # save model
    save_path = saver.save(sess, 'path/model_name.ckpt')

file_writer.flush()
file_writer.close()
print(best_theta)

# restoring a model
with tf.Session() as sess:
    saver.restore(sess, 'path/model_name.ckpt')


# Modularity & Sharing Variables
def relu(X):
    threshold = tf.get_variable('threshold', shape=(),
                                initializer=tf.constant_initializer(0.0))

    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name='weights')
    b = tf.Variable(0.0, name='bias')
    z = tf.add(tf.matmul(X, w), b, name='z')
    return tf.maximum(z, 0.0, name='relu')


n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
relus = []
for relu_index in range(5):
    with tf.variable_scope('relu', reuse=(relu_index >= 1)) as scope:
        relus.append(relu(X))

output = tf.add_n(relus, name='output')


