# There are 2 main types of object in a tf graph, variables and placeholders
# During the optimization process of the model tf tunes the parameters of the model
# Variables can hold the valeus of weights and biases
# These need to be initialized


# Placeholders are initially empty and are used to feed in the training examples in small batches
# They need expected data type and shape of the data

import tensorflow as tf

my_tensor = tf.random_uniform((4,4), minval = 0, maxval= 1)
print my_tensor

my_variable = tf.Variable(initial_value=my_tensor)
# We need to initialize the variable
init = tf.global_variables_initializer()

ph = tf.placeholder(dtype = tf.float32) # also specify shape fx = (Samples of data,features ?)




with tf.Session() as sess:
        print sess.run(my_tensor)
        print "\n"
        sess.run(init)
        print sess.run(my_variable)
