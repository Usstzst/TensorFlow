"""
Chapter13: Convolutional Neural Network
-- TensorFlow Implementation
--


"""

from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

china = load_sample_image('china.jpg')
flower = load_sample_image('flower.jpg')
datasets = np.array([china, flower], dtype=np.float32)
batch_size, height, width, channels = datasets.shape

# Create 3 filters
filters = np.zeros(shape=(7,7,channels,2), dtype=np.float32)
filters[:,3,:,0] = 1 # vertical line
filters[3,:,:,1] = 1 # horizontal line

# create a graph with input X plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
# convolution layer
#convolution = tf.nn.conv2d(X, filters, strides=[1,2,2,1], padding='SAME')

# max pooling layer tf.nn.max_pool function
# you can use tf.nn.avg_pool function too
max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

with tf.Session() as sess:
    # convolution
    #output = sess.run(convolution, feed_dict={X:datasets})
    # pooling layer
    output = sess.run(max_pool, feed_dict={X:datasets})

plt.imshow(output[0,:,:,1], cmap='gray')
plt.show()
