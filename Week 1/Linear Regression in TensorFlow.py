"""

Linear Regression in TensorFlow

"""
# In[1] Imports
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import tensorflow.compat.v1 as tf  # Using TF1
print(tf.__version__)
tf.disable_v2_behavior()  # Disable TF2.0 warnings
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 6)

# In[2]

X = np.arange(0.0, 5.0, 0.1)
X

a = 1
b = -2
Y = a * X + b

# Plot
plt.plot(X, Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

# Lets use Fuel Consumption Dataset
file = r'C:\Users\Desktop\Desktop\Artificial Intelligence Main\IBM AI Engineering\Building Deep Learning Models with TensorFlow\Week 1\FuelConsumptionCo2.csv'
df = pd.read_csv(file)
df.head(10)

# Lets use Linear Regression to predict Co2 Emission based on vehicle engine
train_x = np.asanyarray(df[['ENGINESIZE']])
train_y = np.asanyarray(df[['CO2EMISSIONS']])

# Lets manual initialize variables a and b (or W, b)
a = tf.Variable(20.0)
b = tf.Variable(30.2)
y = a*train_x + b

# Define a loss function for the linear regression model
# Define mean squared error loss
loss = tf.reduce_mean(tf.square(y - train_y))

# Define optimizer
optimizer = tf.train.GradientDescentOptimizer(0.05)

# Train
train = optimizer.minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Initialization of the training and loss list
loss_values = []
train_data = []

for step in range(100):
    _, loss_val, a_val, b_val = sess.run([train, loss, a, b])
    loss_values.append(loss_val)
    if step % 5 == 0:
        print(step, loss_val, a_val, b_val)
        train_data.append([a_val, b_val])

# Plot loss_values
plt.plot(loss_values, 'ro')
plt.ylabel('Loss')
plt.xlabel("Interation")
plt.show()

cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(train_x)
    line = plt.plot(train_x, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(train_x, train_y, 'ro')
green_line = mpatches.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line])
plt.show()


