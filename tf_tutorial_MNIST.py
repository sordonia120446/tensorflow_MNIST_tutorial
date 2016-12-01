"""
Tutorial implementation of tensorflow.  

Link:  https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html

Created on Nov 30, 2016

@author: sordonia120446
"""


import tensorflow as tf

# Importing MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""
Helper function(s) below:
"""

def print_prediction_accuracy():
	# Output prediction vs actual
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # returns list of booleans
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # determine fraction of correct predictions
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


"""
Placeholder = value that we'll input when running a computation. 
None --> dimensions can be any length
784 --> # of dimensions of the vector
"""
x = tf.placeholder(tf.float32, [None, 784])

"""
A Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations. 
Notice that weight W has a shape of [784, 10] because we want to multiply the 784-dimensional image 
vectors by it to produce 10-dimensional vectors of evidence for the difference classes. 
Bias b has a shape of [10] so we can add it to the output.
"""
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define the model for softmax regression
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Training the model

# Implementing cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

"""
Minimize cross-entropy with Gradient descent procedure. 
Gradient descent is a simple procedure, where TensorFlow simply shifts each variable a little bit in the direction that reduces the cost. 
What TensorFlow actually does here, behind the scenes, is it adds new operations to your graph which implement backpropagation and gradient descent. 
Then it gives you back a single operation which, when run, will do a step of gradient descent training, slightly tweaking 
your variables to reduce the cost.
"""
learning_rate = 0.5
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Initialize variables
init = tf.initialize_all_variables()

# Starting session
sess = tf.Session()
sess.run(init)

training_steps = 1000
for i in range(training_steps):
	"""
	Each step of the loop, we get a "batch" of one hundred random data points from our training set. 
	We run train_step feeding in the batches data to replace the placeholders.
	"""
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	if (i % (training_steps/10) == 0):
		predicted_value = tf.argmax(y,1)
		actual_value = tf.argmax(y_,1)
		print_prediction_accuracy()

# After done
print("We're done!  See below for resulting accuracy")
print_prediction_accuracy()
































