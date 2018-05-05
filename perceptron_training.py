# -*- coding: utf-8 -*-
"""
Created on Fri May  4 16:59:47 2018

@author: victzuan
"""

import computational_graphs as cg
import binary_data_generator as bdg
import numpy as np
import matplotlib.pyplot as plt

def generate_new_data(size = 50):
    ##Assume that we are given a dataset consisting of 100 points in the plane. 
    ##Half of the points are red and half of the points are blue.
    
    ##Create 50 red points centered at (-2,-2).
    red_data = bdg.Points("red", size, [-2,-2])
    ##Create 50 blue points centered at (2,2).
    blue_data = bdg.Points('blue', size, [2,2])
    return blue_data, red_data


#Create a new graph.
cg.Graph().as_default()

X = cg.placeholder()
c = cg.placeholder()

#Initialize weights randomly.
W = cg.Variable(np.random.randn(2,2))
b = cg.Variable(np.random.randn(2))

#Build perceptron.
p = cg.softmax(cg.add(cg.matmul(X, W), b))

#Build cross-entropy loss.
J = cg.negative(cg.reduce_sum(cg.reduce_sum(cg.multiply(c, cg.log(p)), axis = 1)))

#Build minimization op.
minimization_op = cg.GradientDescentOptimizer(learning_rate = 0.01).minimizer(J)

blue_data, red_data = generate_new_data()

#Build placeholder inputs.
feed_dict = {
        X: np.concatenate((blue_data.points, red_data.points)),
        c:
            [[1, 0]]*len(blue_data.points)
            +[[0 , 1]]*len(red_data.points)
            }
#Create session.
session = cg.Session()

#Perform 100 gradient descent steps.
for step in range(100):
    J_value = session.run(J, feed_dict)
    if step % 10 == 0:
        print("Step:", step, "Loss:", J_value)
    session.run(minimization_op, feed_dict)

#Print final result
W_value = session.run(W)
print("Weight matrix: \n", W_value)
b_value = session.run(b)
print("Bias:\n", b_value)

#Plont a line.
x_axis = np.linspace(-4,4,100)
y_axis = -W_value[0][0]/W_value[1][0]*x_axis - b_value[0]/W_value[1][0]
plt.plot(x_axis, y_axis)

#Add the red and blue points.
plt.scatter(red_data.points[:,0], red_data.points[:,1], color = 'red')
plt.scatter(blue_data.points[:,0], blue_data.points[:,1], color = 'blue')
plt.show()
    
