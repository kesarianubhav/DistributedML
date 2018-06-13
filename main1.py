import uuid
import os
# from keras.models import Sequential
# from keras.layers import Dense
import numpy as np
import math
import redis
from redis import Redis
from rq import Queue , Connection , Worker
from flask import jsonify
from qr import PriorityQueue
import json

os.system('sudo service redis start')

PORT = 6379
HOST = 'localhost'
prq =PriorityQueue('q',host=HOST,port=PORT)

"""


1. Model Creator
2. Calculation Distributor
3. Graph Creator
"""
def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__


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


def bfs(final_node):
    #print("chutiye"+str(final_node))
    layer =1
    q = []
    for i in final_node.inputs :
        q.append(i)


    # print(len(q))

    while(len(q)!=0):
        element=q.pop()
        if isinstance(element,Variable):
            break
        elif isinstance(element,Node):
            for i in element.inputs:
                q.append(i)
            #print("Node->"+str(element.get_output().value))
        layer+=1

    return layer


def sigmoid(x):
    if isinstance(x, Variable):
        return 1 / (1 + math.exp(-x.value))
    elif isinstance(x,numpy.ndarray):
        return 1 / (1 + np.exp(-x))
    else:
        return 1/(1 + math.exp(-x))


class Node(object):
    def __init__(self, inputs, activation):
        self.inputs = inputs
        self.output = Variable(None)
        np.random.seed(0)
        self.weights = np.random.rand(len(inputs))*(0.01)
        self.activation = activation
        print(self.weights)
        self.id = str(uuid.uuid4())
        self.error = None
        self.status = 'waiting'
        # self.break_calc()
        # self.layer = None
        self.layer=bfs(self)


    def activate(self):
        self.output = self.activation(self.output)

    def compute(self):

        operation = '+'

        if ( isinstance(input,Variable) and i.value ==None for i in self.inputs) or ( isinstance(input,Variable) and i.get_output().value==None for i in self.inputs) :
            print("Cant be computed \n")
        else:
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
                    print("Type Error")

            print(self.output)
            self.activate()
            self.output = Variable(self.output)

    def get_output(self):
        return self.output

    def get_layer(self):
        return self.layer

    def break_calc(self):
        inputs = []
        for inputw in self.inputs:
            if isinstance(inputw, Variable):
                inputs.append(inputw)
            elif isinstance(inputw, Node):
                inputs.append(inputw.get_output())
            else:
                inputs.append(inputw)


        # p ={}
        # p['layer'] =self.layer
        # p['node_id']=self.id
        # p['inputs'] = self.inputs
        # p['operation'] ='+'
        # p['weights'] =self.weights.tolist()
        return {'layer':self.layer,'node_id':self.id, 'inputs': inputs, 'operation': '+', 'weights': self.weights.tolist()}
        # return p

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

    def get_nodes(self):
        return self.nodes


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
        return self.calculations

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


def bubble_down(g):
    for i in g.get_nodes():
        #print(type(i.break_calc))
        #print(json.dumps(dict(i.break_calc())))
        if i.get_output()==None:
            prq.push(i.break_calc(),i.get_layer())

def priority_heap(calculations):
    pass


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
#node1.compute()
#node1.compute()
#node2.compute()
#node3.compute()
print("Node 1's output :" +str(node1.get_output().value))
print("Node 2's output :" +str(node2.get_output().value))
print("Node 3's output :" +str(node3.get_output().value))
#
# print("Node 3's layer :" +str(node3.layer))
# print("Node 3's layer :" +str(node1.layer))



#bfs(node3)
# import json
# json.dumps(node1.break_calc())
# print(isinstance(np.array([10, 20]), np.ndarray))



#total_calculations = (graph.compile())
#for i in enumerate(total_calculations):
    #print(type(i))
    #print("YEH HAI->>> "+str(i['node_id']))
    #i=jsonify(i)

# bubble_down(graph)
print(len(prq))



#
# def get_a_calculation():
#     return graph.calculations[0]
#
#
# print(get_a_calculation())
os.system('sudo service redis stop')
