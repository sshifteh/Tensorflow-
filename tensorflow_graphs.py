# Graphs are connected nodes, vertices and connection is called edges
# Each node is an operation with input and gives output
# In general we will construct a graph in tf and execute it

# We will create a graph with two input nodes which goes into an add operation and give the output of the operation

import tensorflow as tf

node1 = tf.constant(1)
node2 = tf.constant(2)
node3 = node1+node2

with tf.Session() as sess:
        print sess.run(node3)

# When starting tf a default graph is created
graph1= tf.get_default_graph()
print graph1
# new graph
graph2 = tf.Graph()
print graph2

# set g2 as default graph

with graph2.as_default():
        print graph2 is tf.get_default_graph()

# We get True. But this is inside the session. Else graph1 is the default graph
# Important to make sure to ust one graph

