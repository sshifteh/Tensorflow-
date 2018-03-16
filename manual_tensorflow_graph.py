        
# Basically manually creating what tensorflow is doing under the hood
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Operation():
        def __init__(self, input_nodes = []):
            self.input_nodes = input_nodes
            self.output_nodes = []

            for node in input_nodes:
                    node.output_nodes.append(self)

            _default_graph.operation.appen(self)

            def compute(self):
                    # placeholder method
                    pass



class add(Operation):

        def __init__(self, x,y):
            super([x,y])

        def compute(self, xvar, yvar):
            self.inputs = [xvar, yvar]
            return xvar + yvar


class multiply(Operation):

        def __init__(self, x,y):
            super([x,y])

        def compute(self, xvar, yvar):
            self.inputs = [xvar, yvar]
            return xvar * yvar


class matmul(Operation):

        def __init__(self, x,y):
            super([x,y])

        def compute(self, xvar, yvar):
            self.inputs = [xvar, yvar]
            return xvar.dot(yvar)




# Placeholder is an empty node that needs values to give an output
# Variables are changeable parameters of the graph, i.e. the weights of the NN
# Graph - are global variables connecting variables and placeholders to operations

class Placeholder():

        def __init__(self):
            self.output_nodes= []

            # everytime a placeholder is created it is appended to the list of placeholders in the graph class
            _default_graph.placeholders.append(self)


class Variable():
        def __init__(self, initial_value = None):
                self.value = initial_value
                self.output_nodes = []

                _default_graph.variables.append(self)

class Graph():
        def __init__(self):
            self.operations =[]
            self.placeholders = []
            self.variables = []

        def set_as_default(self):

            # global variable so we can access it in variable class etc and in operation class fx
            global _default_graph
            _default_graph = self


# We want to solve using our graph:
# z = Ax +b; A = 10; b =1



# obs ! We need a Post order tree traversal, i.e. the order is important
# Then we make a session to execute everything.

def traverse_postorder(operation):
        nodes_postorder = []
        def recurse(node):
                if isinstance(node, Operation):
                    for input_node in node.input_nodes:
                            recurse(input_node)

        recurse(operation)
        return nodes_postorder




class Session():

        def run(self, operation, feed_dict={}):
            # use feed dict to feed placeholder a value
            # later we will be feeding our network batches of data
            # so we will be feeding the network a batch through that dictionary

             nodes_postorder = traverse_postorder(operation)

             for node in nodes_postorder:

                    if type(node) == Placeholder:
                        node.output = feed_dict[node]
                    elif type(node) == Variable:
                        node.output = node.value
                    else:
                        # then it is an operation
                        node.inputs = [input_node.output for input_node in node.input_nodes]
                        node.output = node.compute(*node.inputs) # asterics for args
                    if type(node.output) == list:
                            node.output = np.array(node.output)

             return operation.output


g = Graph()
g.set_as_default()
x = Placeholder()
A = Variable(10)
b = Variable(1)
#FIXME something is wrong in the inheritance from the super class maybe
#y = multiply([A,x])
#z = add(y,b)

#sess = Session()
#result = sess.run(operation = z, feed_dict={x:10})
#print result


# Try different example
g = Graph()
g.set_as_default()
A = Variable(([10,20],[30,40]))
b = Variable([1,2])
x = Placeholder()
y = matmul(A,x)
z = add(y,b)
sess = Session()
sess.run(operation = z, feed_dict={x:10})

# FIXME Also for this example an error in the inheritance line

# activation function
def sigmoid(z):
        return 1/ (1 + np.exp(-z))

x = np.linspace(-10,10,100)
y = sigmoid(x)

plt.plot(x, y)

# thinks of an activation as an operation so we make it into an operation
class sigmoid():
        def __init__(self,z):
            super([z])

        def compute(self,z_values):
            return 1 / (1 + np.exp(-z))



data = make_blobs(n_samples = 50, n_features=2, centers = 2, random_state=75)
type(data)





