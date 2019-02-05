import tensorflow as tf

# ----------print hello world:

hello = tf.constant("Hello world!")

sess = tf.Session()
print(sess.run(hello))


with tf.Session() as sess:
    print(f"Saying hello: {sess.run(hello)}")

# ---------more basic operations:
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations
add = tf.add(a, b)
mul = tf.multiply(a, b)

# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))


# Matrix Multiplication:
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matrix1, matrix2)


with tf.Session() as sess:
    result = sess.run(product)
    print(result)



# --------- using eager API(To make this work restart kernel and only execute the following eager api block):
"""
eager api is to be able to interact with the object of tensorflow.
for example declaring a as a constant:
    without eager api: when print it it will return Tensor("Const:0", shape=(), dtype=string)
    but with eager it will return the value in it.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Set Eager API
print("Setting Eager mode...")
tfe.enable_eager_execution()

# Define constant tensors
print("Define constant tensors")
a = tf.constant(2)
print("a = %i" % a)
b = tf.constant(3)
print("b = %i" % b)

# Run the operation without the need for tf.Session
print("Running operations, without tf.Session")
c = a + b
print("a + b = %i" % c)
d = a * b
print("a * b = %i" % d)


# Full compatibility with Numpy
print("Mixing operations with Tensors and Numpy Arrays")

# Define constant tensors
a = tf.constant([[2., 1.],[1., 0.]], dtype=tf.float32)
print("Tensor:\n a = %s" % a)
b = np.array([[3., 0.],[5., 1.]], dtype=np.float32)
print("NumpyArray:\n b = %s" % b)

# Run the operation without the need for tf.Session
print("Running operations, without tf.Session")

c = a + b
print("a + b = %s" % c)

d = tf.matmul(a, b)
print("a * b = %s" % d)

print("Iterate through Tensor 'a':")
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])