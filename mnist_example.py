from Valley import sequential, layers 
from Valley.sequential import Sequential 
from Valley.layers import Linear, Tanh, ReLU, Sigmoid, Softmax, MSE

import numpy as np
import csv 

nn = Sequential()
nn.add(Linear(784, 100))
nn.add(Tanh((100)))
nn.add(Linear(100, 10))
nn.add(Sigmoid(10))

Loss = MSE((10))

X = []
y = []
with open('train.csv', 'rb') as csvfile:
    read = csv.reader(csvfile, delimiter = ' ', quotechar = '|')
    counter = 0
    for row in read:
        if counter == 0:
            pass
        
        else:
            X.append(np.asarray(map(int, row[0][2:].split(','))))
            onehot = np.zeros(10)
            onehot[int(row[0][0])] = 1
            y.append(onehot)
        
        counter += 1

test_y = np.asarray(y[:10000])
test_X = np.asarray(X[:10000])

y = np.asarray(y[10000:])
X = np.asarray(X[10000:])

import time 

def create_conf_matrix(answers, targets):
    a = np.zeros((10, 10))
    answers = np.argmax(answers, axis = 1)
    targets = np.argmax(targets, axis = 1)

    for i in range (len(answers)): 
        a[answers[i]][targets[i]] += 1 
    
    return a.astype(int)

for epoch in range (50):
    avg_acc= 0 
    avg_loss = 0
    for point in range (0, len(X), 100):
        output = nn.forward(X[point:point+100])
        output_arg = np.argmax(output, axis = 1)
        #print output_arg
        avg_acc += np.mean(output_arg == np.argmax(y[point:point+100], axis = 1))

        loss = Loss.forward(output, y[point:point+100])
        avg_loss += loss
        grad = Loss.backward(y[point:point+100])
        #print nn.layers[-1].gradients_w
        #print grad
        nn.backward(grad)
        nn.update(100, 0.3/(2 ** (epoch/20)))
    
    print "Epoch", epoch + 1, " | lr", 0.3/(2 ** (epoch/20)), " | Avg acc -", round(avg_acc/(len(X)/100), 3), " | Avg loss -", round(avg_loss/(len(X)/100), 3)

print ""
print "Testing..."
output = nn.forward(test_X)
conf= create_conf_matrix(output, test_y)
print conf
