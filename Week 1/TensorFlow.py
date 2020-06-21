"""

Introducing TensorFlow:

TensorFlow defines computations as Graphs, and these are made with operations (also know as “ops”).
So, when we work with TensorFlow, it is the same as defining a series of operations in a Graph.

To execute these operations as computations, we must launch the Graph into a Session.
The session translates and passes the operations represented into the graphs to the device you want to execute them on, be it a GPU or CPU.
In fact, TensorFlow's capability to execute the code on different devices such as CPUs and GPUs is a consequence of it's specific structure.

"""

import tensorflow as tf
import tensorflow.compat.v1 as tf

# TensorFlow works as a graph computational model. Let's create a graph which we will be named graph1.
graph1 = tf.Graph()

# Construct tf.Operation (nodes) and tf.Tensor (edge) to add them to the graph
with graph1.as_default():
    a = tf.constant([2], name='constant_a')
    b = tf.constant([3], name='constant_b')

# Run a tensorflow session to print values of a and b
sess = tf.Session(graph=graph1)  # tf.Session is deprecated. tf.compat.v1.Session must be used
result = sess.run(a)
result1 = sess.run(b)
print(result, result1)
sess.close()

# Make an operation over tensors
with graph1.as_default():
    c = tf.add(a, b)

# Create a session and print
sess = tf.Session(graph=graph1)
result = sess.run(c)
print(result)
sess.close()

"""
Defining Multidimensional arrays using TensorFlow
Define Constants (Cannot be updated)
"""

graph2 = tf.Graph()
with graph2.as_default():
    Scalar = tf.constant(2)
    Vector = tf.constant([5, 6, 2])
    Matrix = tf.constant([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    Tensor = tf.constant([[[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[4, 5, 6], [5, 6, 7], [6, 7, 8]], [[7, 8, 9], [8, 9, 10], [9, 10, 11]]])
with tf.Session(graph=graph2) as sess:
    result = sess.run(Scalar)
    print("Scalar (1 entry):\n %s \n" % result)
    result = sess.run(Vector)
    print("Vector (3 entries) :\n %s \n" % result)
    result = sess.run(Matrix)
    print("Matrix (3x3 entries):\n %s \n" % result)
    result = sess.run(Tensor)
    print("Tensor (3x3x3 entries) :\n %s \n" % result)

# Check shapes of data created
print("Shape of Scalar: ", Scalar.shape)
print("Shape of Vector: ", Vector.shape)
print("Shape of Matrix: ", Matrix.shape)
print("Shape of Tensor: ", Tensor.shape)

# Lets perform some addition on tensors
graph3 = tf.Graph()
with graph3.as_default():
    Matrix_one = tf.constant([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    Matrix_two = tf.constant([[0, -2, -3], [-2, -2, -4], [-3, -4, -4]])

    add_1_operation = tf.add(Matrix_one, Matrix_two)
    add_2_operation = Matrix_one + Matrix_two

with tf.Session(graph=graph3) as sess:
    result = sess.run(add_1_operation)
    print("Defined using tensorflow function :")
    print(result)
    result = sess.run(add_2_operation)
    print("Defined using normal expressions :")
    print(result)

# Lets perform matrix multiplication on tensors - using matmul
graph4 = tf.Graph()
with graph4.as_default():
    Matrix_one = tf.constant([[2, 3], [3, 4]])
    Matrix_two = tf.constant([[2, 3], [3, 4]])

    mul_operation = tf.matmul(Matrix_one, Matrix_two)
with tf.Session(graph=graph4) as sess:
    result = sess.run(mul_operation)
    print("Defined using tensorflow function:")
    print(result)


"""
Define Variables. Unlike constants, variables can be updated each run through multiple sessions.
"""

# start a session to run the graph and initialize then update
graph5 = tf.Graph()
with graph5.as_default():
    # Define variable v
    v = tf.Variable(0)
    # Define the update that will add +1 to the variable
    update = tf.assign(v, v+1)
    # Variables must be initialized be running an initialization operating after launching the graph.
    init_op = tf.global_variables_initializer()

with tf.Session(graph=graph5) as session:
    session.run(init_op)
    print(session.run(v))
    for i in range(3):
        session.run(update)
        print(session.run(v))

"""
Placeholders. Used to feed data from outside the TensorFlow graph into the graph.
Placeholders can be seen as "holes" in the model, "holes" data must be passed into.
tf.placeholder(datatype),
where datatype specifies the type of data (integers, floating points, strings, booleans)
along with its precision (8, 16, 32, 64) bits.
"""

graph6 = tf.Graph()
with graph6.as_default():
    a = tf.placeholder(tf.float32)  # Empty placeholder with data type float32
    b = a*2

with tf.Session(graph=graph6) as session:
    result = session.run(b, feed_dict={a: 3.5})  # Feeding float value 3.5 to placeholder a
    print(result)

dictionary = {a: [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]]}

with tf.Session(graph=graph6) as session:
    result = session.run(b, dictionary)  # Feeding dictionary with tensor
    print(result)


"""
Operations are nodes that represent the mathematical operations over the tensors on a graph.
These operations can be any kind of functions, like add and subtract tensor or maybe an activation function.
tf.constant, tf.matmul, tf.add, tf.nn.sigmoid are some of the operations in TensorFlow.
"""

graph7 = tf.Graph()
with graph7.as_default():
    a = tf.constant([5])
    b = tf.constant([2])
    c = tf.add(a, b)
    d = tf.subtract(a, b)

with tf.Session(graph=graph7) as session:
    result = session.run(c)
    print('c =: %s' % result)
    result = session.run(d)
    print('d =: %s' % result)
