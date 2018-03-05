import tensorflow as tf
import numpy as np
import pandas as pd


df = pd.read_csv('/home/chris/Documents/git_repos_github/titanic_survivors/tensorflow/trainmod.csv')    
df = df.drop(['Unnamed: 0'],axis = 1)
x_train = df.iloc[:,2:15]
n = len(x_train.columns) # X is an m by n matrix
x_train = x_train.as_matrix()
m = len(x_train)
y_train= df.iloc[:,1]
y_train = y_train.as_matrix()

datapoint_size = m

batch_size = datapoint_size
# batch_size: Configure this to:
#             1: stochastic mode
#             integer < datapoint_size: mini-batch mode
#             datapoint_size: batch mode
# i: Current epoch number







x = tf.placeholder(tf.float32, [m, n])
theta = tf.Variable(tf.zeros([n,1]))
b = tf.Variable(tf.zeros([m,1]))
y = tf.matmul(x,theta) + b # this is m by 1 matrix
h = 1 / (1 + tf.exp(-y)) # this needs to be done pointwise as y is m by 1, np.exp should be able to handle it.

y_ = tf.placeholder(tf.float32, [m, 1]) 

cost = -y_ * tf.log(h) - (1 - y) * tf.log(1 - h) # the ones here are not the correct length. They need to be m by 1.

alpha = 0.02
train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
steps = 1000
for i in range(steps):

    if datapoint_size == batch_size:
  # Batch mode so select all points starting from index 0
      batch_start_idx = 0
      batch_end_idx = batch_start_idx + batch_size
      batch_xs = x_train[batch_start_idx:batch_end_idx]
      batch_ys = y_train[batch_start_idx:batch_end_idx]
 #   elif datapoint_size < batch_size:
  # Not possible
#      raise ValueError(“datapoint_size %d, must be greater than         
  #                  batch_size: %d” % (datapoint_size, batch_size))
    else:
  # stochastic/mini-batch mode: Select datapoints in batches
  #                             from all possible datapoints
      batch_start_idx = (i * batch_size) % (datapoint_size - batch_size)
      batch_end_idx = batch_start_idx + batch_size
      batch_xs = x_train[batch_start_idx:batch_end_idx]
      batch_ys = y_train[batch_start_idx:batch_end_idx]

# Get batched datapoints into xs, ys, which is fed into
# 'train_step'
    xs = np.array(batch_xs)
    ys = np.array(batch_ys)

    feed = { x: xs, y_: ys }
    sess.run(train_step, feed_dict=feed)
    print("After %d iteration:" % i)
    print("theta: %f" % sess.run(theta))
    print("b: %f" % sess.run(b))
    print("cost: %f" % sess.run(cost, feed_dict=feed))
