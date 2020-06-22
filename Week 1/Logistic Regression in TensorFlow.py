"""

Logistic Regression in TensorFlow

"""
# In[1] Imports
import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
print(tf.__version__)
tf.disable_v2_behavior()  # Disable TF2.0 warnings

# In[2] Dataset

# For this logitic regression exercise the Iris dataset (inbuilt) will be used

# Load dataset
iris = load_iris()
iris_X, iris_y = iris.data[:-1, :], iris.target[:-1]
iris_y = pd.get_dummies(iris_y).values  # Converts data into (1, 0, 0) or (0, 1, 0) or (0, 0, 1) for 3 classes

# Split data using sklearn train_test_split
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=0)

# Create placeholders for the model
# But first we need to know the shape the placeholders will be
# numFeatures is number of features in the input data.
# iris dataset numFeatures is 4
numFeatures = trainX.shape[1]
numFeatures

# numLabels is number of classes in the dataset, which is 3
numLabels = trainY.shape[1]
numLabels

"""Placeholders
"""

# None is set to tell TensorFlow that there isn't a specific number for the dimension
X = tf.placeholder(tf.float32, [None, numFeatures])
yGold = tf.placeholder(tf.float32, [None, numLabels])

"""Parameters
"""

# Initialize model weights and biases to zeros
# TrainX dimension is (training examples, features)
# W Variable should be in the shape (features, number of nodes)
# B Variable shape should be (number of nodes)
W = tf.Variable(tf.zeros([4, 3]))
n = tf.Variable(tf.zeros[3])

W = tf.Variable(tf.random_normal([numFeatures, numLabels],
                                 mean=0,
                                 stddev=0.01,
                                 name='weights'))
b = tf.Variable(tf.random_normal([numLabels],
                                 mean=0,
                                 stddev=0.01,
                                 name='bias'))


"""Operators
"""
# Breakdown Logistic regression yhat = sigmoid(WX + b) into 3 parts
apply_weights_OP = tf.matmul(X, W, name='apply_weights')
add_bias_OP = tf.add(apply_weights_OP, b, name='add_bias')
activation_OP = tf.nn.sigmoid(add_bias_OP, name='activation')


"""Training
"""
numEpochs = 20000

# Define learning rate
learning_rate = tf.train.exponential_decay(learning_rate=0.0001,
                                           global_step=1,
                                           decay_steps=trainX.shape[0],
                                           decay_rate=0.95,
                                           staircase=True)

# Define cost function
cost_OP = tf.nn.l2_loss(activation_OP - yGold, name='squared_error_cost')  # Not optimal cost function

# Train
training_OP = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_OP)


"""Running Operations
"""

sess = tf.Session()  # Create TensorFlow session (Not a thing in TF2)

# Initialize the weights and biases variables
init_OP = tf.global_variables_initializer()

# run session
sess.run(init_OP)


"""More Operations
"""
# argmax(activation_OP, 1) returns the label with the most probability
# argmax(yGold, 1) is the correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP, 1), tf.argmax(yGold, 1))

# If every false prediction is 0 and every true prediction is 1, the average returns us the accuracy
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

# Summary OP for regression output
activation_summary_OP = tf.summary.histogram('ouput', activation_OP)

# Summary OP for accuracy
accuracy_summary_OP = tf.summary.scalar('accuracy', accuracy_OP)

# Summary op for cost
cost_summary_OP = tf.summary.scalar("cost", cost_OP)

# Summary ops to check how variables (W, b) are updating after each iteration
weightSummary = tf.summary.histogram("weights", W.eval(session=sess))
biasSummary = tf.summary.histogram("biases", b.eval(session=sess))

# Merge all summaries
merged = tf.summary.merge([activation_summary_OP, accuracy_summary_OP, cost_summary_OP, weightSummary, biasSummary])

# Summary writer
writer = tf.summary.FileWriter("summary_logs", sess.graph)

"""
Training Loop
"""
cost = 0
diff = 1
epoch_values = []
accuracy_values = []
cost_values = []

for i in range(numEpochs):
    if i > 1 and diff < 0.00001:
        print("Change in cost %g; Convergence." %diff)  # When convergence happens
        break

    else:
        # Run training
        step = sess.run(training_OP, feed_dict={X: trainX, yGold: trainY})
        # Report occasional stats every 10 iterations
        if i % 10 == 0:
            epoch_values.append(i)
            # Generate accuracy stats on test data
            train_accuracy, newCost = sess.run([accuracy_OP, cost_OP], feed_dict={X: trainX, yGold: trainY})
            # Add accuracy to live graphing variable
            accuracy_values.append(train_accuracy)
            # Add cost to live graphinh variable
            cost_values.append(newCost)
            # Re-assign values for variabes
            diff = abs(newCost - cost)
            cost = newCost

            # Generate print statements
            print("step %d, training accuracy %g, cost %g, change in cost %g"%(i, train_accuracy, newCost, diff))

# How well do we perform on held-out test data?
print("final accuracy on test set: %s" %str(sess.run(accuracy_OP,
                                                     feed_dict={X: testX,
                                                                yGold: testY})))


# Ploting
plt.plot([np.mean(cost_values[i-50:i]) for i in range(len(cost_values))])
plt.show()

