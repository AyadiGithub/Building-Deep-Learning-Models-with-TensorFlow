
"""

CONVOLUTIONAL NEURAL NETWORK APPLICATION

"""
# In[1] Imports

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
print(tf.__version__)


# In[2] Import dataset MNIST

#Start interactive session
sess = tf.InteractiveSession()

# Load mnist dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[3] Setting up CONV Layer 1

# Create placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Reshape x for convolutional layer
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Assigning weights and bias to zeroes
# Kernel (5,5) with 1 channel (grayscale) with 32 feature maps (32 filters)
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))  # 32 biases for 32 outputs

# Create convolutional layer and relu
convolve1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')+b_conv1

# Relu applied to convolve1
h_conv1 = tf.nn.relu(convolve1)

# Maxpool
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv1


# In[4] CONV Layer 2

# Assigning weights and bias to zeroes
# Input from previous layer is 14x14x32 (Due to maxpooling and filters)
# Filter size should be 5x5x32, kernel size (5,5)
# The output will be 14x14x64
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))  # 64 biases for 64 outputs

# Convolutional layer with input from maxpooling layer conv1
convolve2 = tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')+b_conv2

# Relu
h_conv2 = tf.nn.relu(convolve2)

# Maxpool
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  #max pool 2x2
conv2
# Shape after 2nd max pooling is 7x7x64

# In[5] Flatten and Fully Connected Layer and dropout

# We need to flatten the 7x7x64 output to 1 column
layer2_matrix = tf.reshape(conv2, [-1, 7*7*64])
layer2_matrix

# Weights and biases for fully connected layer of size 1024
W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

# Matrix multiplication
fc1 = tf.matmul(layer2_matrix, W_fc1) + b_fc1
fc1

# Relu
h_fc1 = tf.nn.relu(fc1)
h_fc1

# Lets implement a dropout layer to prevent overfitting
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)
layer_drop

# In[6] Output layer with softmax

# Input is 1024 and output is 10
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

# Matrix multiplciation of dropout layer by W_fc2 weight
fc = tf.matmul(layer_drop, W_fc2)

# Softmax applied on fc layer (output)
y_CNN = tf.nn.softmax(fc)
y_CNN

# In[7] Setup loss function cross_entropy, optimizer, Accuracy

# Cross_entropy loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))

# Optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Define Correct Prediction
correct_prediction = tf.equal(tf.argmax(y_CNN, 1), tf.argmax(y_, 1))

# Define Accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize varaibles
sess.run(tf.global_variables_initializer())

for i in range(1100):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


# In[8] Evaluate Model
# evaluate in batches to avoid out-of-memory issues
n_batches = mnist.test.images.shape[0] // 50
cumulative_accuracy = 0.0
for index in range(n_batches):
    batch = mnist.test.next_batch(50)
    cumulative_accuracy += accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
print("test accuracy {}".format(cumulative_accuracy / n_batches))

# Lets see the filters
kernels = sess.run(tf.reshape(tf.transpose(W_conv1, perm=[2, 3, 0,1]),[32, -1]))



