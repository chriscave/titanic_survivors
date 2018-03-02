import tensorflow as tf
import numpy as np
import pandas as pd


df = pd.read_csv('/home/chris/Documents/git_repos_github/titanic_survivors/tensorflow/trainmod.csv')    

X = df
Y = df last column

[m n] = size(X)

x = tf.placeholder(tf.float, [m, n])
theta = tf.Variable(tf.zeros([n,1]))
b = tf.Variable(tf.zeros[1])
y = tf.matmul(x,theta) + b # this is m by 1 matrix
h = 1 / (1 + np.exp(-y)) # this needs to be done pointwise as y is m by 1, np.exp should be able to handle it.

y_ = tf.placeholder(tf.float32, [None, m]) 

cost = -y_ * np.log(h) - (1 - y) * np.log(1 - h) # the ones here are not the correct length. They need to be m by 1.

alpha = 0.02
train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
steps = 1000
for i in range(steps):
  feed = { x: X, y_: Y }
  sess.run(train_step, feed_dict=feed)
  print("After %d iteration:" % i)
  print("W: %f" % sess.run(W))
  print("b: %f" % sess.run(b))
  print("cost: %f" % sess.run(cost, feed_dict=feed))
