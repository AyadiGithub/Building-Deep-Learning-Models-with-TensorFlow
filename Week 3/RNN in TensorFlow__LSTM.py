"""

RECURRENT NETWORKS IN DEEP LEARNING (LSTM)

- The way an LSTM works is simple: you have a linear unit, which is the information cell itself,
surrounded by three logistic gates responsible for maintaining the data.
One gate is for inputting data into the information cell, one is for outputting data from the input cell,
and the last one is to keep or forget data depending on the needs of the network.

- The Long Short-Term Memory is composed of a linear unit surrounded by three logistic gates:

the "Input" or "Write" Gate, which handles the writing of data into the information cell
the "Output" or "Read" Gate, which handles the sending of data back onto the Recurrent Network
the "Keep" or "Forget" Gate, which handles the maintaining and modification of the data stored in the information cell

A usual flow of operations for the LSTM unit is as such:
    1) The Keep Gate has to decide whether to keep or forget the data currently stored in memory
    It receives both the input and the state of the Recurrent Network, and passes it through its Sigmoid activation.

  #  𝐾𝑡 (Keep Gate) = 𝜎( 𝑊𝑘 * [𝑆𝑡−1, 𝑥𝑡] + 𝐵𝑘)

    If 𝐾𝑡 has value of 1 means that the LSTM unit should keep the data stored perfectly
    and if 𝐾𝑡 a value of 0 means that it should forget it entirely.

 #   𝑂𝑙𝑑𝑡 = 𝐾𝑡 * 𝑂𝑙𝑑𝑡−1  [𝑂𝑙𝑑𝑡−1 is the data previously in memory.]

    2) The input and state are passed on to the Input Gate, in which there is another Sigmoid activation applied.

  #  𝐼𝑡 (Input Gate) = 𝜎 (𝑊𝑖 * [𝑆𝑡−1,𝑥𝑡] + 𝐵𝑖)

    𝑁𝑒𝑤𝑡 (New data to be input into the memory cell)
    𝐶𝑡 is result of the processing of the inputs by the Recurrent Network

 #   𝑁𝑒𝑤𝑡 = 𝐼𝑡 * 𝐶𝑡

   𝑁𝑒𝑤𝑡 is the new data to be input into the memory cell.
   This is then added to whatever value is still stored in memory.

 #   𝐶𝑒𝑙𝑙𝑡 = 𝑂𝑙𝑑𝑡 + 𝑁𝑒𝑤𝑡

   3) The Output Gate functions in a similar manner.
    To decide what we should output, we take the input data and state and pass it through a Sigmoid function as usual.
    The contents of our memory cell, however, are pushed onto a Tanh function to bind them between a value of -1 to 1.
    Consider 𝑊𝑜 and 𝐵𝑜 as the weight and bias for the Output Gate.

  #  𝑂𝑡 = 𝜎 (𝑊𝑜 * [𝑆𝑡−1,𝑥𝑡] + 𝐵𝑜)

  Outputt is the output into the Recurrent Network

  # 𝑂𝑢𝑡𝑝𝑢𝑡𝑡 = 𝑂𝑡 * 𝑡𝑎𝑛ℎ(𝐶𝑒𝑙𝑙𝑡)

"""

# In[1] Imports to create LSTM

import numpy as np
import tensorflow as tf
sess = tf.Session()

# In[2] 1 LSTM cell

# Create LSTM network with only 1 cell
# We have to pass 2 elements to LSTM, the prv_output and prv_state, so called, h and c.
# Therefore, we initialize a state vector, state.
# Here, state is a tuple with 2 elements, each one is of size [1 x 4],
# one for passing prv_output to next time step, and another for passing the prv_state to next time stamp.

LSTM_CELL_SIZE = 4  # Output size (dimension) - Same as the hidden size 4
lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
# 2 * state of `1,4
state = (tf.zeros([1, LSTM_CELL_SIZE]), ) * 2
state

# Define sample unit. Batch_size = 1 and Seq_len = 6
sample_input = tf.constant([[3, 2, 2, 2, 2, 2]], dtype=tf.float32)
print(sess.run(sample_input))

# Pass input to lstm cell and check new state
with tf.variable_scope("LSTM_sample1"):
    output, state_new = lstm_cell(sample_input, state)
sess.run(tf.global_variables_initializer())
print(sess.run(state_new))  # We see there are two parts: new state (c) and output (h)
sess.close()

# In[3] Stacked LSTM

sess = tf.Session()

input_dim = 6

# Stacked LSTM cell
cells = []

# 1st Layer
LSTM_CELL_SIZE_1 = 4  # 4 hidden nodes
cell1 = tf.contrib.rnn.LSTMCell(LSTM_CELL_SIZE_1)
cells.append(cell1)

# 2nd Layer
LSTM_CELL_SIZE_2 = 5  # 5 hidden nodes
cell2 = tf.contrib.rnn.LSTMCell(LSTM_CELL_SIZE_2)
cells.append(cell2)

# Define the stacked LSTM
stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells)

"""
Create the RNN from stacked_lstm
"""
# batch_size x time steps x features
data = tf.placeholder(tf.float32, [None, None, input_dim])
output, state = tf.nn.dynamic_rnn(stacked_lstm, data, dtype=tf.float32)

# Lets try input sequence length as 3 and dimensionality as 6
# Input tensor should be of shape [batch_size, max_time, dimension]  -- (2, 3, 6)
sample_input = [[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],[[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]]
sample_input

output

# Run session
sess.run(tf.global_variables_initializer())
sess.run(output, feed_dict={data: sample_input})




















