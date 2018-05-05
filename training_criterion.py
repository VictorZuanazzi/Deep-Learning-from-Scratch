# -*- coding: utf-8 -*-
"""
Created on Wed May  2 07:59:12 2018

@author: VICTZUAN
source: http://www.deepideas.net/deep-learning-from-scratch-iii-training-criterion/
"""
import computational_graphs as cg
import binary_data_generator as bdg
import numpy as np

def generate_new_data(size = 50):
    ##Assume that we are given a dataset consisting of 100 points in the plane. 
    ##Half of the points are red and half of the points are blue.
    
    ##Create 50 red points centered at (-2,-2).
    red_data = bdg.Points("red", size, [-2,-2])
    ##Create 50 blue points centered at (2,2).
    blue_data = bdg.Points('blue', size, [2,2])
    ## Plot the red and blue points.
    red_data.plot_points()
    blue_data.plot_points()
    return blue_data, red_data

def example_line(w = -1, b=0):
    ##Apparently, we can draw a line y=−x that nicely separates the space into a red 
    ##region and a blue region:
    division_line = bdg.Line(w,b)
    ##Plot a line y=-x.
    division_line.plot_line([-4,4])
    return division_line


def example():
    #Create a new graph.
    cg.Graph().as_default()
    X = cg.placeholder()
    c = cg.placeholder()
    
    #Create a weight matrix for 2 output classes:
    #One with a weight vector [1,1] for blue and one witha weight vector [-1, -1]
    #for red.
    W = cg.Variable([
            [1, -1],
            [1, -1]
            ])
    b = cg.Variable([0,0])
    p = cg.softmax(cg.add(cg.matmul(X, W), b))
    
    #An alternative is to use maximum likelihood estimation, where we try to 
    #find the parameters that maximize the probability of the training data:
    J = cg.negative(cg.reduce_sum(cg.reduce_sum(cg.multiply(c, cg.log(p)), axis=1)))
    
    blue_data, red_data = generate_new_data()
    division_line = example_line()
    
    #Create a session and run the perceptrion on our blue/red points.
    session = cg.Session()
    output_probabilities = session.run(J,{
            X: np.concatenate((blue_data.points, red_data.points)),
            c: 
                [[1, 0]]*len(blue_data.points)
                +[[0, 1]]*len(red_data.points)
            })
    

    print(output_probabilities)

example()
