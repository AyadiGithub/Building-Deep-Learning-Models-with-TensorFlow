"""

Autoencoders with TensorFlow

"""

# In[1] Imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# In[2] Model parameters

learning_rate = 0.001
training_epochs = 30
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters (3 layer encoder, 3 layer decoder)
n_hidden_1 = 256  # 1st layer number of features
n_hidden_2 = 128  # 2nd layer number of features
n_hidden_3 = 64  # 3rd layer
n_input = 784  # 28x28 images

# tf graph input (images)
X = tf.placeholder("float", [None, n_input])

# Initialize weights dictionary
weights = {'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
           'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
           'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
           'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
           'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
           'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

# Initialize biase dictionary
biases = {'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
          'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
          'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
          'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
          'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
          'decoder_b3': tf.Variable(tf.random_normal([n_input])),
}

# In[3] Create Encoder and Decoder


def encoder(x):
    # Encoder 1st layer
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Encoder 2nd layer
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    # Encoder 3rd layer
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))

    return layer_3


def decoder(x):
    # Decoder 1st layer
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder 2nd layer
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    # Decoder 3rd layer
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))

    return layer_3


# In[4] Initialize model

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Reconstructed Images
y_pred = decoder_op

# Labels (Targets) or the input data X
y_true = X

# Define loss (MSE) and Optimizer (RMSProp)
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initialize variables
init = tf.global_variables_initializer()

# In[5] Training
sess.close()
sess = tf.InteractiveSession()
sess.run(init)

total_batch = int(mnist.train.num_examples / batch_size)

# Training cycle
for epoch in range(training_epochs):
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Run backprop with optimizer and calculate cost with cost function
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})

    # Display logs per epoch
    if epoch % display_step == 0:
        print("Epoch: ", '%04d' % (epoch+1), "cost = ", "{:.9f}".format(c))

print("Training Finished")

# In[6] Evaluation
encode_decode = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})

# Plots for visualization
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))












