# -*- coding: utf-8 -*-
"""
Created on Sun May  6 13:00:46 2018

@author: VICTZUAN
source: http://www.deepideas.net/deep-learning-from-scratch-v-multi-layer-perceptrons/
"""
import computational_graphs as cg
import binary_data_generator as bdg
import numpy as np
import matplotlib.pyplot as plt

def generate_linear_data(size = 50):
    ##Assume that we are given a dataset consisting of 100 points in the plane. 
    ##Half of the points are red and half of the points are blue.
    
    ##Create 50 red points centered at (-2,-2).
    red_data = bdg.Points("red", size, [-2,-2])
    ##Create 50 blue points centered at (2,2).
    blue_data = bdg.Points('blue', size, [2,2])
    return blue_data, red_data


def generate_data(center = [[-2, -2],[2, 2]], size = 50):
    data = []
    for j in range(size):
        for i in center:
            data.append(np.random.randn(2)+i)
    
    return data 

def example():
    #Create new graph.
    cg.Graph().as_default()
    
    #Create training input placeholder.
    X = cg.placeholder()
    
    #Create placeholder for the training classes.
    c = cg.placeholder()
    
    #Build a hidden layer 
    W_hidden = cg.Variable(np.random.randn(2,2))
    b_hidden = cg.Variable(np.random.randn(2))
    p_hidden = cg.sigmoid(cg.add(cg.matmul(X, W_hidden), b_hidden))
    
    #Build the output layer.
    W_output = cg.Variable(np.random.randn(2,2))
    b_output = cg.Variable(np.random.randn(2))
    p_output = cg.softmax(cg.add(cg.matmul(p_hidden, W_output), b_output))
    
    #Build cross-entropy loss.
    J = cg.negative(cg.reduce_sum(cg.reduce_sum(cg.multiply(c, cg.log(p_output)), axis = 1)))
    
    #Build minimization op.
    minimization_op = cg.GradientDescentOptimizer(learning_rate = 0.03).minimizer(J)
    blue_data, red_data = generate_linear_data()
    #Build placeholder inputs.
    feed_dict = {
            X: np.concatenate((blue_data.points, red_data.points)),
            c:
                [[1, 0]]*len(blue_data.points)
                +[[0, 1]]*len(red_data.points)
                }
    
    #Create session().
    session = cg.Session()
    
    #Perform 1000 gradient descent steps
    for step in range(1000):
        J_value = session.run(J, feed_dict)
        if step%100 == 0:
            print("Step: ", step, " Loss: ", J_value)
        session.run(minimization_op, feed_dict)
        
    #Print final result.
    W_hidden_value = session.run(W_hidden)
    print ("Hidden layer weight matrix: \n", W_hidden_value)
    b_hidden_value = session.run(b_hidden)
    print ("Hidden layer bias: \n", b_hidden_value)
    W_output_value = session.run(W_output)
    print ("Output layer weight matrix: \n", W_output_value)
    b_output_value = session.run(b_output)
    print ("Output layer bias: \n", b_output_value)
    
    xs = np.linspace(-2,2)
    ys = np.linspace(-2,2)
    pred_classes = []
    for x in xs:
        for y in ys:
            pred_class = session.run(p_output,
                                     feed_dict = {X: [[x, y]]})[0]
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
    
    plt.plot(xs_p, ys_p, 'ro', xs_n, ys_n, 'bo')


def example_2():
    #Create new graph.
    cg.Graph().as_default()
    
    #Create training input placeholder.
    X = cg.placeholder()
    
    #Create placeholder for the training classes.
    c = cg.placeholder()
    
    #Build a hidden layer 
    W_hidden = cg.Variable(np.random.randn(2,2))
    b_hidden = cg.Variable(np.random.randn(2))
    p_hidden = cg.sigmoid(cg.add(cg.matmul(X, W_hidden), b_hidden))
    
    #Build the output layer.
    W_output = cg.Variable(np.random.randn(2,2))
    b_output = cg.Variable(np.random.randn(2))
    p_output = cg.softmax(cg.add(cg.matmul(p_hidden, W_output), b_output))
    
    #Build cross-entropy loss.
    J = cg.negative(cg.reduce_sum(cg.reduce_sum(cg.multiply(c, cg.log(p_output)), axis = 1)))
    
    #Build minimization op.
    minimization_op = cg.GradientDescentOptimizer(learning_rate = 0.03).minimizer(J)
    blue_data = generate_data(center=[[-2, -2],[2, 2],[2, -2]])
    red_data = generate_data(center=[[-2, 2]])
    #plt.scatter(blue_data[:,0], blue_data[:,1], color= "blue")
    #plt.scatter(red_data[:,0], red_data[:,1], color= "red")
    #Build placeholder inputs.
    feed_dict = {
            X: np.concatenate((blue_data, red_data)),
            c:
                [[1, 0]]*len(blue_data)
                +[[0, 1]]*len(red_data)
                }
    
    #Create session().
    session = cg.Session()
    
    #Perform 1000 gradient descent steps
    for step in range(1000):
        J_value = session.run(J, feed_dict)
        if step%100 == 0:
            print("Step: ", step, " Loss: ", J_value)
        session.run(minimization_op, feed_dict)
        
    #Print final result.
    W_hidden_value = session.run(W_hidden)
    print ("Hidden layer weight matrix: \n", W_hidden_value)
    b_hidden_value = session.run(b_hidden)
    print ("Hidden layer bias: \n", b_hidden_value)
    W_output_value = session.run(W_output)
    print ("Output layer weight matrix: \n", W_output_value)
    b_output_value = session.run(b_output)
    print ("Output layer bias: \n", b_output_value)
    
    xs = np.linspace(-4,4)
    ys = np.linspace(-4,4)
    pred_classes = []
    for x in xs:
        for y in ys:
            pred_class = session.run(p_output,
                                     feed_dict = {X: [[x, y]]})[0]
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
    
    plt.plot(xs_p, ys_p, 'ro', xs_n, ys_n, 'bo')
    
    
    
    
example_2()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    