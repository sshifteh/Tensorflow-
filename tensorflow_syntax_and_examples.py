import tensorflow as tf
print(tf.__version__)

# tensor is an n dim array
# most basic is a constant

var = tf.constant("Hello")
var2 = tf.constant("World")
print type(var)
# this is a tensor object, we do not get the variable hello string
print var
# to get hello to print we need to run it inside a session

# with automatically opens it and closes it
with tf.Session() as sess:
        # everything inside is tensorflow op that we run
        result = sess.run(var+var2)

print result

# perform addition computation
a = tf.constant(10)
b = tf.constant(20)
print type(a)
print a +b

with tf.Session() as sess:
        result = sess.run(a+b)

print result

# Example 3:
const= tf.constant(10)
fill_mat = tf.fill((4,4,), 10)
myzeros = tf.zeros((4,4,))
ones = tf.ones((4,4))
myrandn= tf.random_normal((4,4), mean = 0, stddev = 1.0)
myranduniform = tf.random_uniform((4,4), minval = 0, maxval = 1)

my_operations = [const, fill_mat, myzeros, ones, myrandn, myranduniform]
# FIXME this syntax did not work. But its not that important.
#sess = tf.InteractiveSession()
#for operation in my_operations:
#        sess.run(operation)

with tf.Session() as sess:
    for operation in my_operations:
            print sess.run(operation)
            print ("\n")


# matrix mul:
A = tf.constant([[1,2], [3,4] ])
print A.get_shape()

B = tf.constant([[10], [100]])

with tf.Session() as sess:
        result = tf.matmul(A,B)
        sess.run(result)
print result
# FIXME this should have printed a 2 by 1 - matrix
