import uuid
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import math

"""

1. Model Creator
2. Calculation Distributor
3. Graph Creator
"""

# 1. Model Creator
#
# model = Sequential()
# model.add(Dense(10, input_shape=(None, 15)))
# model.add(Dense(5))
# model.add(Dense(3))
# model.compile(optimizer='adam', loss='mse')
#
# x_shape = (32, 15)
# total_calculations = 10
#
#
# print(model.summary())

# 2. Calculations Distributor
# X = np.array([[1, 2, 3, 4], [3, 4, 5, 4], [5, 6, 7, 4]])
# y = np.array([1, 0, 1])
#
# W1 = np.array([[2, 5, 6, 5],
#                [3, 5, 7, 6],
#                [9, 2, 4, 8]])
#
# W2 = np.array([[3],
#                [4],
#                [6]])
#
# print(X.shape, W1.shape)
#
# print(np.dot(W1.T, X))


def sigmoid(x):
    if isinstance(x, Variable):
        return 1 / (1 + math.exp(-x.value))
    else:
        return 1 / (1 + math.exp(-x))


class Node(object):
    def __init__(self, inputs, activation):
        self.inputs = inputs
        self.output = Variable(None)
        self.weights = np.random.rand(len(inputs))
        self.activation = activation
        print(self.weights)
        self.id = str(uuid.uuid4())
        self.error = None
        self.status = 'waiting'

        self.break_calc()

    def compute(self):

        operation = '+'

        for index, input in enumerate(self.inputs):
            if isinstance(input, Variable):

                if self.output.value is None:
                    self.output.value = input.value * self.weights[index]
                else:
                    self.output.value += input.value * self.weights[index]
            elif isinstance(input, Node):
                if self.output.value is None:
                    self.output.value = input.get_output().value * self.weights[index]
                else:
                    self.output.value += input.get_output().value * self.weights[index]
            else:
                print("Wrong input type")

        print(self.output)
        self.activate()
        self.output = Variable(self.output)

    def get_output(self):
        return self.output

    def activate(self):
        self.output = self.activation(self.output)

    def break_calc(self):
        inputs = []
        for input in self.inputs:
            if isinstance(input, Variable):
                inputs.append(input)
            elif isinstance(input, Node):
                inputs.append(input.get_output())
            else:
                inputs.append(input)

        return {'node_id':self.id, 'inputs': inputs, 'operation': '+', 'weights': self.weights.tolist()}


class Layer(object):
    def __init__(self):
        self.nodes = []
        self.output = None

    def add(self, node):
        self.nodes.append(node)

    def get_output(self):
        self.output = [node.get_output() for node in self.nodes]


class Graph(object):
    def __init__(self):
        self.nodes = []
        self.output = None
        self.calculations = []

    def add(self, node):
        self.nodes.append(node)

    def __str__(self):
        return "Graph"

    def get_output(self):
        return self.output

    def run(self):
        for node in self.nodes:

            # node.compute()

            if node.status == 'waiting':

                for input_node in node.inputs:

                    if not isinstance(input_node, Variable):
                        print(input_node.status)
                        if input_node.status != 'calculated':
                            print("Has to wait for inputs")
                        else:
                            print("Heelll")

                self.calculations.append(node.break_calc())

        self.output = self.nodes[-1].get_output()

    def compile(self):
        for node in self.nodes:
            self.calculations.append(node.break_calc())


class CalculationGraph(object):
    def __init__(self):
        self.nodes = []

    def add(self, node):
        self.nodes.append(node)


class Variable(object):
    def __init__(self, value):
        self.value = value
        self.id = str(uuid.uuid4())

    # def __str__(self):
    #     return "{}".format(self.value)


graph = Graph()

var1 = Variable(10)
var2 = Variable(20)
var3 = Variable(10)
var4 = Variable(40)

node1 = Node([var1, var2, var3, var4], sigmoid)
node2 = Node([var1, var2, var3, var4], sigmoid)

node3 = Node([node1, node2], sigmoid)

graph.add(node1)
graph.add(node2)
graph.add(node3)

# layer = Layer()
# layer.add(node1)

# graph.run()

# print(graph.get_output())

#print(node1.break_calc()['inputs'][0].value)

node1.compute()
node2.compute()

print("Node 1 output:", node1.get_output().value)
print("Node 2 output:", node2.get_output().value)

# print(node2.break_calc()['inputs'][0].value)
#
# print(node3.break_calc()['inputs'][0].value)

# import json
# json.dumps(node1.break_calc())
# print(isinstance(np.array([10, 20]), np.ndarray))

print(graph.compile())


def get_a_calculation():
    return graph.calculations[0]


print(get_a_calculation())
