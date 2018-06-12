import uuid
import numpy as np
from main import Variable


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
                print("Type Error")

        print(self.output)
        self.activate()
        self.output = Variable(self.output)

    def get_output(self):
        return self.output

    def activate(self):
        self.output = self.activation(self.output)

    def break_calc(self):
        inputs = []
        for inputw in self.inputs:
            if isinstance(inputw, Variable):
                inputs.append(inputw)
            elif isinstance(inputw, Node):
                inputs.append(inputw.get_output())
            else:
                inputs.append(inputw)

        return {'node_id':self.id, 'inputs': inputs, 'operation': '+', 'weights': self.weights.tolist()}
