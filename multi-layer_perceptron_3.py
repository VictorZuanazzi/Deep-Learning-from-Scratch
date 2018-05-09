# -*- coding: utf-8 -*-
"""
Created on Tue May  8 06:22:45 2018

@author: VICTZUAN
"""

import computational_graphs as cg
import numpy as np
import matplotlib.pyplot as plt

#Neural net is built under the neural_net class.
class neural_net():
    
    def __init__(self, number_of_layers):
        '''Inicializes all the parameters of the neural net.
        
        Args:
            layers: number of perceptron layers. (Only one perceptron per layer?)
            X: cg.placeholder for the computational graph.
            w: list of matrix of perceptrons weights.
            b: list of perceptrons bias.
            p: the perceptrons.
        '''
        #Create a graph where all the magic will happen.
        cg.Graph().as_default()
        #Create training input placeholder.
        self.X = cg.placeholder()
        #Create placeholder for the training classes.
        self.c = cg.placeholder()
        self.layers = number_of_layers
    
        #Randomly set the parameters w, b and learning_rate
        self.w = self.generate_weights()
        self.b = self.generate_bias()
        self.learning_rate = (np.random.randn()**2)/1000
        #Inicializes the perceptron.
        self.p = self.create_perceptron()
         #Create session().
        self.session = cg.Session()
        

    def generate_weights(self):
        '''Generate random weights for the perceptron.
            Return a list of matrix of weights.
        '''
        return list(map(lambda x: cg.Variable(np.random.randn(2,2)), range(self.layers)))
    
    def parameters_from_data(self, data):
        W = []
        for i in range(self.layers):
            w=[]
            w.append(data[np.random.randint(len(data))])
            w.append(data[np.random.randint(len(data))])
            W.append(w)
            
        self.w = list(map(lambda x: 
            cg.Variable(W[x]), 
            range(self.layers)))
        self.b= list(map(lambda x: 
            cg.Variable(data[np.random.randint(len(data))]), 
            range(self.layers)))
    
    def generate_bias(self):
        '''Generate random bias for the perceptron.
            Return a list of arrays of bias.
        '''
        return list(map(lambda x: cg.Variable(np.random.randn(2)), range(self.layers)))
    
    def create_perceptron(self):
        '''It inicializes the perceptron.
            Return a list of perceptrons. Each one of them is alone in its layer.
        '''
        p = []
        p.append(cg.sigmoid(cg.add(cg.matmul(self.X, self.w[0]), self.b[0])))
        #Build other layers
        for i in range(1, self.layers):
            p.append(cg.softmax(cg.add(cg.matmul(p[i-1], self.w[i]), self.b[i])))
        return p
    
    def cross_entropy_loss(self):
        return cg.negative(cg.reduce_sum(cg.reduce_sum(cg.multiply(self.c, cg.log(self.p[-1])), axis = 1)))

    def minimization_op(self, J):
        return cg.GradientDescentOptimizer(learning_rate= self.learning_rate).minimizer(J)
        
    def train(self, data_1, data_2, learning_rate = 0, epochs = 1000):
        if learning_rate != 0:
            self.learning_rate = learning_rate

        #Build cross-entropy loss.
        J = self.cross_entropy_loss()

        #Build minimization op.
        minimization_op = self.minimization_op(J)
        
        #Build placeholder inputs.
        feed_dict = {
                self.X: np.concatenate((data_1, data_2)),
                self.c:
                    [[1, 0]]*len(blue_data)
                    +[[0, 1]]*len(red_data)
                    }

        #Perform 1000 gradient descent steps
        for step in range(epochs):
            J_value = self.session.run(J, feed_dict)
            if step%100 == 0:
                print("Step: ", step, " Loss: ", J_value)
            self.session.run(minimization_op, feed_dict)
    
    def print_perceptron(self):
        print ("learning rate: ", self.learning_rate)
        for i in range(self.layers):
            print("Perceptron layer: ", i)
            print("W[", i, "]:\n", 
                  self.session.run(self.w[i]),
                  "\nbias[",i,"]:",
                  self.session.run(self.b[i]))
            
    def plot_perceptron(self):
        xs = np.linspace(-4,4)
        ys = np.linspace(-4,4)
        pred_classes = []
        for x in xs:
            for y in ys:
                pred_class = self.session.run(self.p[-1],
                                         feed_dict = {self.X: [[x, y]]})[0]
                pred_classes.append((x,y, pred_class.argmax()))
        xs_p, ys_p = [], []
        xs_n, ys_n = [], []
        for x, y, c in pred_classes:
            if c == 0:
                xs_n.append(x)
                ys_n.append(y)
            else:
                xs_p.append(x)
                ys_p.append(y)
        
        plt.plot(xs_p, ys_p, 'rx', xs_n, ys_n, 'bo')
    
def generate_data(center = [[-2, -2],[2, 2]], size = 50):
    data = []
    for j in range(size):
        for i in center:
            data.append(np.random.randn(2)+i)
    
    return data   

def generate_lin_data(center = [[-2, -2],[2, 2]], size = 50):
    data = []
    for j in range(size):
        for i in center:
            data.append((np.random.randn(2)+i)*(1+(2*j)/size))
    
    return data  
#Creates the data.
blue_data = generate_data(center=[[-2, -2],[2, 2],[2, -2]])
red_data = generate_data(center=[[-2, 2], [0,0]])
#
#blue_data = generate_lin_data(center=[[2, -2],[4, 0]])
#red_data = generate_lin_data(center=[[-2, 2], [-4,0]])

#Build layers:
numb_layers = 2

red_blue = neural_net(numb_layers)

#Reset the weights and bias to random data points.
red_blue.parameters_from_data(blue_data + red_data)
#Train the neural net.
red_blue.train(blue_data, red_data, learning_rate = 0.03)

#Print final result.
red_blue.print_perceptron()
        
red_blue.plot_perceptron()





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    