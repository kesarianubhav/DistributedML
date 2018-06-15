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
from collections import defaultdict

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


# def validate_last_gradients(W1,X,W2,Y):
#
#     Z1 = np.dot(W1,X)
#     A1 = 1/(1+np.exp(-Z1))
#     Z2 = np.dot(W2,A1)
#     A2 = 1/(1+np.exp(-Z2))
#     J = np.sum((A2 - Y)*(A2-Y)*0.5)
#     dA2 = (A2-Y)
#     dZ2 = np.dot(dA2 , A1(1-A1).T)
#     dW2 = np.dot(dZ2,A1.T)
#
#     return dW2


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


def sig_derivative_wrt(z,x):
    return z*(1-z)*x


def mse(a,y):
    return float((a-y)*(a-y))*0.5

def mae(a,y):
    return abs(a-y)

def mse_derivative(a,y,x):
    return ((a-y)*x)

class Node(object):
    def __init__(self, inputs, activation):
        self.inputs = inputs
        self.output = Variable(None)
        self.weights = np.random.rand(len(inputs))*(0.01)
        self.activation = activation
        self.id = str(uuid.uuid4())
        self.gradients = np.zeros((len(inputs)))
        self.status = 'waiting'
        self.layer=bfs(self)
        self.error = 0
        # print(self.weights)
        # self.layer = None
        # self.break_calc()

    def activate(self):
        self.output = self.activation(self.output)

    def compute(self):

        operation = '+'
        flag=1

        for i in self.inputs:
            if ( isinstance(i,Variable) and i.value==None) or (isinstance(i,Node) and i.get_output().value==None):
                flag=0
                break



        if flag==1:
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

            #print(self.output)
            self.activate()
            self.output = Variable(self.output)
            self.status = 'calculated'

    def get_output(self):
        # print ( type(self.output))
        return self.output


    def set_output(self,value):
        a = Variable(value )
        self.output= Variable(value)
        return a

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
#
# class Edge(object):
#     self.input=None
#     self.output=None
#
#     def __init__(self):
#         self.weight = list(np.random.randn(1,1))
#     #
#     # def set_input(self,a):
#     #     self.input=a
#     #
#     # def set_output(self,a):
#     #     self.output=a
#
#     def set_weight(self,a):
#         self.weight=a
#
#     def connect(self,a,b):
#         self.input = a
#         sefl.output =b
#



class Graph(object):
    def __init__(self):
        self.nodes = []
        self.output = None
        self.calculations = []

    def add(self, node):
        self.nodes.append(node)

        # def add(self, node):
        #     self.nodes.append(node)

    def __str__(self):
        return "Graph"

    def get_output(self):
        return self.output

    def get_nodes(self):
        return self.nodes

    def get_node_by_id(node_id):
        for i in self.nodes:
            if i.id == node_id:
                return i

    def run(self):
        for node in self.nodes:

            if node.status == 'waiting':

                for input_node in node.inputs:

                    if not isinstance(input_node, Variable):
                        print(input_node.status)
                        if input_node.status != 'calculated':
                            print("Has to wait for inputs")
                        else:
                            print("Calculated")

                self.calculations.append(node.break_calc())

        self.output = self.nodes[-1].get_output()

    def compile(self):
        for node in self.nodes:
            self.calculations.append(node.break_calc())
        return self.calculations

    def forward_propogation(self):
        for node in self.nodes:
            node.compute()

    def last_layer_nodes(self):
        last_layer_nodes = []
        #print(len(self.nodes))
        a = (self.nodes[len(self.nodes)-1])
        #print(a.get_layer())
        for i in self.nodes:
            if (i.get_layer() == a.get_layer()):
                last_layer_nodes.append(i)

        return last_layer_nodes

    def last_error(self,node,y):
        a = node.get_output().value
        # print(type(a))
        # print(type(y))
        dE_da = (a - y)
        da_dz = ( dE_da ) * a*(1-a)
        return da_dz

    def error(self,node):
        da_dz = node.get_output().value * (1-node.get_output().value)
        return da_dz

    def node_backprop(self,node):
        for i in range(0,len(node.inputs)):
            if (node.inputs[i].get_output().value==None):
                print("Cant Backprop !! Output Empty")
                break

            node.gradient[i]+=node.output*sig_derivative_wrt(node.output,node.inputs[i].get_output().value)
        for i in node.inputs:
            i.output=Variable(node.gradient[i])

            return node


    # def backpropogation(self,Y_actual):
    #     count = 0
    #     Y_actual.reverse()
    #     print(len(self.nodes))
    #     for i in range(len(self.nodes)-1,-1,-1):
    #         # print("chutiye !!!")
    #         if self.nodes[i].get_output().value==None:
    #             print("Cant Do Backprop !! Forwardprop Still Not Done !!")
    #             break
    #
    #         else:
    #             print("node no= "+str(i))
    #             a = len(Y_actual)
    #             b = len(self.last_layer_nodes())
    #             print(a)
    #             print(b)
    #             assert(a==b) , "Y_ACTUAL AND Y_PREDICT FEATURES DO NOT MATCH "
    #             j=0
    #             if (count <=len(self.last_layer_nodes())):
    #                 self.nodes[i].set_output (Variable(Y_actual[j]-self.nodes[i].get_output().value))
    #                 self.nodes[i]=node_backprop(self.node[i])
    #                 count+=1
    #             else:
    #                 self.nodes[i]=node_backprop(self.nodes[i])
    #             j+=1

    #
    def backpropogation(self,Y_actual):
        t_n = len(self.nodes)
        l_n = len(self.last_layer_nodes())
        for i in range ( t_n-1 , t_n - l_n -1 , -1 ):
            self.nodes[i].error  = self.last_error(self.nodes[i] , Y_actual[0].value)
            for j in range( 0,len(self.nodes[i].weights)):
                self.nodes[i].gradients[j] = self.nodes[i].gradients[j] + self.nodes[i].error * self.nodes[i].inputs[j].get_output().value
            for  j in range ( 0,len(self.nodes[i].inputs)):
                self.nodes[i].inputs[j].error=self.nodes[i].inputs[j].error + self.nodes[i].error*self.nodes[i].weights[j]

        for i in range ( t_n-l_n-1,-1,-1):
            print(i)
            # print(type(self.nodes[i].inputs[0]))
            for j in range(0,len(self.nodes[i].gradients)):
                if isinstance( self.nodes[i].inputs[j],Node):
                    self.nodes[i].gradients[j] = self.nodes[i].error* self.error(self.nodes[i])*self.nodes[i].inputs[j].get_output().value
                if isinstance( self.nodes[i].inputs[j],Variable):
                    self.nodes[i].gradients[j] = self.nodes[i].error* self.error(self.nodes[i])*self.nodes[i].inputs[j].value
            for j in range(0,len(self.nodes[i].inputs)):
                if isinstance( self.nodes[i].inputs[j],Node):
                    self.nodes[i].inputs[j].error = self.nodes[i].inputs[j].error + self.nodes[i].error*self.nodes[i].weights[j]


    def node_update(self,node,learning_rate):
        for j in range ( 0 , len(node.gradients)-1):
            node.weights[j]=node.weights[j]+(learning_rate*node.gradients[j])



    def updation(self,learning_rate):
        for i in self.nodes:
            self.node_update(i,learning_rate)


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


def id_view(g):
    d = defaultdict(list)
    for i in g.nodes:
        for j in i.inputs:
            d[(i.id)].append((j.id))

    return d


def reverse(g):
    dict2 = defaultdict(list)
    view = id_view(g)
    for key in view.keys():
        for i in view[key]:
            dict2[i].append((key))

    for key in dict2.keys():
        for node in g.nodes :
            if(node.id==key):
                node.inputs = dict2[key]
                node.layer+=bfs(node)
                break
    return g


graph = Graph()

f = open("test_data.txt","r")
# var1 = Variable(p[0])
# var2 = Variable(None)
# var3 = Variable(None)
#
# var4 = Variable(None)
#
# for i in f:
#     # print(i)
# p = i.split(" ")
# for j in range(0 , len(p))  :
#     #p[j] = Variable(int(p[j]))
#     p[j] = Variable(float(p[j]))
#     print(p[j].value)
#
#
#
# var1 = p[0]
# var2 = p[1]
# var3 = p[2]
# var4 = p[3]
#
#
# Y_actual=p[4]

np.random.seed(0)

var1 = Variable(1)
var2 = Variable(1)
var3 = Variable(1)
var4 = Variable(1)

Y_actual = [Variable(4)]

node1 = Node([var1, var2, var3, var4], sigmoid)
node2 = Node([var1, var2, var3, var4], sigmoid)

node3 = Node([node1, node2], sigmoid)

graph.add(node1)
graph.add(node2)
graph.add(node3)
#
# node1.compute()
# node2.compute()
# node3.compute()
#
# print(str(node1.weights))
# print(str(node2.weights))
# print(str(node3.weights))
#
graph.forward_propogation()
#
# print("Node 1's output :" +str(node1.get_output().value))
# print("Node 2's output :" +str(node2.get_output().value))
# print("Node 3's output :" +str(node3.get_output().value))
#
graph.backpropogation(Y_actual)
#
# print("Gradient at node1 = "+str(node1.gradients))
# print("Gradient at node2 = "+str(node2.gradients))
# print("Gradient at node3 = "+str(node3.gradients))
#
graph.updation(learning_rate=0.5)

print("Node 1 Weights="+str(node1.weights))
print("Node 2 Weights="+str(node2.weights))
print("Node 3 Weights="+str(node3.weights))


#
# print("PART TWO -VERIFICATION DATA !!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
#
# X =np.array([1,1,1,1]).reshape(4,1)
# # print(X.reshape(4,1))
# W1 = np.vstack((graph.nodes[0].weights , graph.nodes[1].weights))
# # print(W1.shape)
# Z1 = np.dot(W1,X).reshape(2,1)
# # print(Z1.shape)
# A1 = 1/(1+np.exp(-Z1))
# A1 = (A1.reshape(2,1))
# # print("Using numpy Layer 1's output:" +str(A1))
# # print(str(node1.get_output().value) +  str(node1.get_output().value))
# # assert
#
#
# W2 = graph.nodes[2].weights.reshape(1,2)
# Z2 = np.dot(W2,A1)
# A2 = 1 /(1+np.exp(Z2))
# # print("Using numpy Node 3's output:" +str(A2))
# Y=np.array([4])
# J = (np.sum(A2-Y)*(A2-Y)*0.5)
#
#
# dA2 = (A2-Y)
# # print(dA2.shape)
# dZ2 = (dA2 * (A2*(1-A2)))
# dW2 = np.dot(dZ2,A1.T)
# print(dW2)
#
# # print(A1.shape)
# # print(dZ2.shape)
# # print(W2.T.shape)
# dZ1=np.dot(W2.T,dZ2)*A1*(1-A1)
# assert(dZ1.shape==Z1.shape) ,"Shape Mismatch"
# # print(X.T.shape)
# dW1=np.dot(dZ1,X.T)
# print(dW1)
# print(node2.error)
# print(node1.error)
# print(node3.error)
#
# print("\n")
#
# # print(dA2)
# print(dZ2)
# print(dZ1)






# print(len(graph.last_layer_nodes()))
# layer = Layer()
# layer.add(node1)

# graph.run()

# print(graph.get_output())
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
# print(len(prq))
# # print(id_view(graph))
# for i in graph.nodes :
#     print("ID -> "+str(i.id) + " nodes = "+str(i.inputs)+ "\n\n")
#
# print("REVERSED GRAPH")
#
# for i in reverse(graph).nodes :
#     print("ID -> "+str(i.id) + " nodes = "+str(i.inputs)+ "\n\n")






#
# print(id_view(reverse(graph)))

#
# def get_a_calculation():
#     return graph.calculations[0]
#
#
# print(get_a_calculation())
os.system('sudo service redis stop')
