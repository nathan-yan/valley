import numpy as np

class MSE:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.flattened_shape = np.prod(self.input_shape)

        self.gradients_i = None

        self.parameters = False 

        self.inputs = None 
        self.inputs_flattened = None
        self.loss = None
    
    def forward(self, inputs, targets):
        self.inputs = inputs 
        self.loss = np.sum( (inputs - targets) ** 2)/len(targets)
        return self.loss

    def backward(self, targets, inputs = None):
        if inputs == None:
            self.gradients_i = 2 * (self.inputs - targets)/self.flattened_shape
        else:
            self.gradients_i = 2 * (inputs-  targets)/self.flattened_shape
       # print self.inputs
        return self.gradients_i

class Linear: 
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.parameters = True 

        self.gradients_w = None
        self.gradients_i = None
        self.gradients_b = None

        self.inputs = None 
        self.outputs = None 
    
        self.weights = np.random.random((input_shape, output_shape)) - 0.5
        #self.bias = np.random.random((1, output_shape)) - 0.5
        self.bias = np.zeros((1, output_shape))

    def fprop(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.bias
    
    def bprop(self, gradients, inputs = None):
        if inputs == None: 
            self.gradients_w = np.dot(self.inputs.T, gradients)
        else:
            self.gradients_w = np.dot(inputs.T, gradients)

        self.gradients_b = np.sum(gradients)
        self.gradients_i = np.dot(gradients, self.weights.T)

class ReLU:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.flattened_shape = np.prod(self.input_shape)

        self.gradients_i = None 
        
        self.parameters = False 


        self.inputs = None 
        self.outputs = None
    
    def fprop(self, inputs):
        self.inputs = inputs 
        self.outputs = np.maximum(inputs, 0)

    def bprop(self, gradients, inputs = None):
        if inputs == None:
            self.gradients_i = (self.outputs != 0) * gradients 
        else:
            self.gradients_i = (self.inputs > 0) * gradients

class Tanh:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.flattened_shape = np.prod(self.input_shape)

        self.gradients_i = None 

        self.parameters = False 

        self.inputs = None 
        self.outputs = None
    
    def fprop(self, inputs):
        self.inputs = inputs
        self.outputs = np.tanh(self.inputs)

    def bprop(self, gradients, inputs = None):
        if inputs == None:
            self.gradients_i = (1 - self.outputs ** 2) * gradients
        else:
            outputs = np.tanh(inputs)
            self.gradients_i = (1 - outputs ** 2) * gradients