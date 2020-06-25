"""
    Restricted Boltzmann Machine (RBM)

RBM is a 2 layer neural network.
RBMs are shallow neural nets that learn to reconstruct data by themselves in an unsupervised fashion.
They can automatically extract meaningful features from a given input.
RBM takes the inputs and translates them into binary values that represents them in the hidden layer.
They are then translated back to reconstruct the inputs.
Through several forward and backward passes, the RBM will be trained, and a trained RBM can reveal which features are the most important ones when detecting patterns.


An RBM has two layers. The first layer of the RBM is called the visible (or input layer)
The second layer is the hidden layer, which possesses the neurons.

Each node in the first layer also has a bias denoted as “v_bias” for the visible units.
The v_bias is shared among all visible units.

The second layer (hidden) bias is denoted as “h_bias” for the hidden units. The h_bias is shared among all hidden units.

"""

# In[1] Imports

# Load utility file which contains different utility functions that help in processing the outputs into a more understandable way.
import urllib.request
with urllib.request.urlopen("http://deeplearning.net/tutorial/code/utils.py") as url:
    response = url.read()
target = open('utils.py', 'w')
target.write(response.decode('utf-8'))
target.close()

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from utils import tile_raster_images
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (18.0, 18.0)

# In[2] Initialization

# Define shared bias for input and hidden layer
v_bias = tf.placeholder("float", [7])
h_bias = tf.placeholder("float", [2])

# Define weight matrix (7 visible neurons by 2 hidden neurons)
W = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(7, 2)).astype(np.float32))

"""
RBM has two phases:

    Forward Pass
    Backward Pass or Reconstruction

Forward pass: Input one training sample (one image) X through all visible nodes, and pass it to all hidden nodes.
Processing happens in each node in the hidden layer.
This computation begins by making stochastic decisions about whether to transmit that input or not (i.e. to determine the state of each hidden layer).
At the hidden layer's nodes, X is multiplied by a 𝑊𝑖𝑗 and added to h_bias.
The result of those two operations is fed into the sigmoid function, which produces the node’s output, 𝑝(hj), where j is the unit number.

𝑝(hj) = 𝜎(∑𝑖 𝑤𝑖𝑗 * 𝑥𝑖), where 𝜎() is the logistic function.
𝑝(hj) is the probability of the hidden unit and all values together are the probability distribution.



"""


# In[3] Toy example

# Phase 1
sess = tf.Session()
X = tf.constant([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
v_state = X
print ("Input: ", sess.run(v_state))

h_bias = tf.constant([0.1, 0.1])
print ("hb: ", sess.run(h_bias))
print ("w: ", sess.run(W))

# Calculate the probabilities of turning the hidden units on:
h_prob = tf.nn.sigmoid(tf.matmul(v_state, W) + h_bias)  #probabilities of the hidden units
print ("p(h|v): ", sess.run(h_prob))

# Draw samples from the distribution:
h_state = tf.nn.relu(tf.sign(h_prob - tf.random_uniform(tf.shape(h_prob)))) #states
print ("h0 states:", sess.run(h_state))

# Phase 2
vb = tf.constant([0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1])
print ("b: ", sess.run(vb))
v_prob = sess.run(tf.nn.sigmoid(tf.matmul(h_state, tf.transpose(W)) + vb))
print ("p(vi∣h): ", v_prob)
v_state = tf.nn.relu(tf.sign(v_prob - tf.random_uniform(tf.shape(v_prob))))
print ("v probability states: ", sess.run(v_state))

# given current state of hidden units and weights, what is the probability of generating [1. 0. 0. 1. 0. 0. 0.] in reconstruction phase,
# based on the above probability distribution function?

inp = sess.run(X)
print(inp)
print(v_prob[0])
v_probability = 1
for elm, p in zip(inp[0],v_prob[0]) :
    if elm ==1:
        v_probability *= p
    else:
        v_probability *= (1-p)
v_probability

# The network is not trained so it is normal to get low similarity.




# In[4] RBM using MNIST

# Load mnist and assign training and testing data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trainX, trainY, testX, testY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

trainX[1].shape  # 784 pixels. Therefore visible layer must be 784 neurons/nodes

# Create placeholders for data.
vb = tf.placeholder("float", [784])  # 784 (28x28) nodes
hb = tf.placeholder("float", [50])  # 50 nodes hidden layer

W = tf.placeholder("float", [784, 50])  # Weight matrix is (784, 50)

# Define visible Layer
v0_state = tf.placeholder("float", [None, 784])

# Define hidden Layer
h0_prob = tf.nn.sigmoid(tf.matmul(v0_state, W) + hb)  # Probabilities of the hidden units
h0_state = tf.nn.relu(tf.sign(h0_prob - tf.random_uniform(tf.shape(h0_prob)))) # Sample_h_given_X

# Define reconstruction
v1_prob = tf.nn.sigmoid(tf.matmul(h0_state, tf.transpose(W)) + vb)
v1_state = tf.nn.relu(tf.sign(v1_prob - tf.random_uniform(tf.shape(v1_prob)))) #sample_v_given_h

# Define error (MSE)
error = tf.reduce_mean(tf.square(v0_state - v1_state))

# In[5] Model training

"""

In order to train an RBM, we have to maximize the product of probabilities assigned to all rows v (images) in the training set V
# (a matrix, where each row of it is treated as a visible vector v):
Which is equivalent to maximizing the expected log probability of V:
So, we have to update the weights Wij to increase p(v) for all v in our training data during training.
So we have to calculate the derivative: -- This cannot be easily done by typical gradient descent (SGD)
we can use another approach, which has 2 steps:
    Gibbs Sampling
    Contrastive Divergence



"""

h1_prob = tf.nn.sigmoid(tf.matmul(v1_state, W) + hb)
h1_state = tf.nn.relu(tf.sign(h1_prob - tf.random_uniform(tf.shape(h1_prob)))) #sample_h_given_X

alpha = 0.01
W_Delta = tf.matmul(tf.transpose(v0_state), h0_prob) - tf.matmul(tf.transpose(v1_state), h1_prob)
update_w = W + alpha * W_Delta
update_vb = vb + alpha * tf.reduce_mean(v0_state - v1_state, 0)
update_hb = hb + alpha * tf.reduce_mean(h0_state - h1_state, 0)

# Initialize variables
cur_w = np.zeros([784, 50], np.float32)
cur_vb = np.zeros([784], np.float32)
cur_hb = np.zeros([50], np.float32)
prv_w = np.zeros([784, 50], np.float32)
prv_vb = np.zeros([784], np.float32)
prv_hb = np.zeros([50], np.float32)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Error for first run
sess.run(error, feed_dict={v0_state: trainX, W: prv_w, vb: prv_vb, hb: prv_hb})

# Parameters
epochs = 20
batchsize = 100
weights = []
errors = []

for epoch in range(epochs):
    for start, end in zip( range(0, len(trainX), batchsize), range(batchsize, len(trainX), batchsize)):
        batch = trainX[start:end]
        cur_w = sess.run(update_w, feed_dict={v0_state: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0_state: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={v0_state: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
        if start % 10000 == 0:
            errors.append(sess.run(error, feed_dict={v0_state: trainX, W: cur_w, vb: cur_vb, hb: cur_hb}))
            weights.append(cur_w)
    print ('Epoch: %d' % epoch,'reconstruction error: %f' % errors[-1])
plt.plot(errors)
plt.xlabel("Batch Number")
plt.ylabel("Error")
plt.show()

# Final weight after training
uw = weights[-1].T
print (uw) # a weight matrix of shape (50,784)


# In[6] Evaluation

tile_raster_images(X=cur_w.T, img_shape=(28, 28), tile_shape=(5, 10), tile_spacing=(1, 1))
image = Image.fromarray(tile_raster_images(X=cur_w.T, img_shape=(28, 28) ,tile_shape=(5, 10), tile_spacing=(1, 1)))
### Plot image
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')

image = Image.fromarray(tile_raster_images(X =cur_w.T[10:11], img_shape=(28, 28),tile_shape=(1, 1), tile_spacing=(1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (4.0, 4.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')

# Lets look at reconstructed Image
file = r'C:\Users\Desktop\Desktop\Artificial Intelligence Main\IBM AI Engineering\Building Deep Learning Models with TensorFlow\Week 4\destructed3.jpg'
img = Image.open(file)
img

# Pass image through the RBM network
sample_case = np.array(img.convert('I').resize((28,28))).ravel().reshape((1, -1))/255.0

hh0_p = tf.nn.sigmoid(tf.matmul(v0_state, W) + hb)
#hh0_s = tf.nn.relu(tf.sign(hh0_p - tf.random_uniform(tf.shape(hh0_p))))
hh0_s = tf.round(hh0_p)
hh0_p_val,hh0_s_val  = sess.run((hh0_p, hh0_s), feed_dict={ v0_state: sample_case, W: prv_w, hb: prv_hb})
print("Probability nodes in hidden layer:" ,hh0_p_val)
print("activated nodes in hidden layer:" ,hh0_s_val)

# reconstruct
vv1_p = tf.nn.sigmoid(tf.matmul(hh0_s_val, tf.transpose(W)) + vb)
rec_prob = sess.run(vv1_p, feed_dict={ hh0_s: hh0_s_val, W: prv_w, vb: prv_vb})

# Plot reconstruction
img = Image.fromarray(tile_raster_images(X=rec_prob, img_shape=(28, 28),tile_shape=(1, 1), tile_spacing=(1, 1)))
plt.rcParams['figure.figsize'] = (4.0, 4.0)
imgplot = plt.imshow(img)
imgplot.set_cmap('gray')













