from Valley import sequential, layers 
from Valley.sequential import Sequential 
from Valley.layers import Linear, Tanh, ReLU, MSE

import numpy as np

nn = Sequential()
nn.add(Linear(2, 5))
nn.add(Tanh((5)))
nn.add(Linear(5, 5))
nn.add(Tanh((5)))
nn.add(Linear(5, 2))

Loss = MSE((2))

X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])

import time 
for epoch in range (1000):
    for point in range (len(X)):
        output = nn.forward(X)

        loss = Loss.forward(output, y)
        print "Loss:", loss
        grad = Loss.backward(y)
        #print nn.layers[-1].gradients_w
        #print grad
        nn.backward(grad)
        nn.update(4, 0.01)

for point in range (len(X)):
    print "Evaluating point", X[point]
    print "Expected result", y[point]
    print "Network output", nn.forward(X[point: point + 1])
    print ""

"""print loss
gradient = Loss.backward(np.array([[2, 4]]))
print gradient
"""
#input_vec = np.random.random(size = (1, 2))
#gradient_vec = np.asarray([[1]])

#nn.forward(input_vec)
#nn.backward(gradient_vec)
"""
grads = nn.layers[0].gradients_w

epsilon = 1e-4
for weight in range (4):
    if weight != 10000: 
        nn.layers[0].weights[weight/2][weight%2] += epsilon
        nn.forward(input_vec)
        o1 = np.sum(nn.layers[2].outputs)
        nn.layers[0].weights[weight/2][weight%2] -= 2 * epsilon 
        nn.forward(input_vec)
        o2 = np.sum(nn.layers[2].outputs)
        
        nn.layers[0].weights[weight/2][weight%2] += 2 * epsilon
        print (o1 - o2)/(2 * epsilon), grads[weight/2][weight%2]
"""
