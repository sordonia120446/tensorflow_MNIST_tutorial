"""
Tutorial implementation of tensorflow.  There are many comments below that elaborate each portion of the script.  

Link:  https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html

Created on Nov 30, 2016

@author: sordonia120446
"""


import tensorflow as tf

# Importing MNIST data
"""
Takes in the data, represented as an image and a corresponding digit.  Each written digit is "flattened" into a 28x28 tensor. 

The labels, y-values, represent the images drawn in each datapoint.  We make these labels as "one-hot-vectors."  It's like a Kronicker delta-ish function.  
It is zero in all dimensions except one, where it has value 1.  It can also be thought of as a bit vector.  
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""
Helper function(s) below:
"""

def print_prediction_accuracy():
	# Output prediction vs actual
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # returns list of booleans
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # determine fraction of correct predictions
	accuracy_value = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
	print("The prediction is this accurate:  {}%".format(100*accuracy_value))


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

"""
Define the model for softmax regression.  

The softmax function is a logistic function that "squashes" a K-dimensional vector/tensor of real values into a K-dimensional vector/tensor 
of real values ranging from (0,1) that sum up to 1.  In probability theory, it is used to represent a categorical distribution.  

Example:  

If we take an input of [1,2,3,4,1,2,3], the softmax of that is [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]. 
The output has most of its weight where the '4' was in the original input. 
This is what the function is normally used for: to highlight the largest values and suppress values which are significantly below the maximum value.

In this script file, it takes the form y = Wx + b

For more details, see https://en.wikipedia.org/wiki/Softmax_function
"""
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Training the model

# Implementing cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

"""
Minimize cross-entropy with Gradient descent procedure.  Cross-entropy compares two probability distributions and determines the 
"loss" between the predicated and true datasets.  

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

training_factor = 1
training_steps = training_factor*1000
for i in range(training_steps):
	"""
	Each step of the loop, we get a "batch" of one hundred random data points from our training set. 
	We run train_step feeding in the batches data to replace the placeholders.

	The training "saturates" at ~92% accuracy.  Adding additional steps doesn't improve past this point, and the accuracy oscillates around 91%.  
	"""
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	if (i % (training_steps/10) == 0):
		predicted_value = tf.argmax(y,1)
		actual_value = tf.argmax(y_,1)
		print_prediction_accuracy()

# After done
print("\nWe're done!  See below for resulting accuracy")
print_prediction_accuracy()
































