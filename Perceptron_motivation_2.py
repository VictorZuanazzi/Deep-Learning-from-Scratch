# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 19:15:48 2018

@author: victzuan
Source: http://www.deepideas.net/deep-learning-from-scratch-ii-perceptrons/
"""
import numpy as np
import matplotlib.pyplot as plt

class Points():
   
    def __init__(self, color = "white", number_of_points = 50, center = [0,0]):
        self.color = color
        self.center = center
        self.nop = number_of_points
        #Randomly creates points normaly distributed around the center.
        self.points = np.random.randn(self.nop, 2) + self.center
        
    def plot_points(self):
        '''Plots the points into a scatter graph.  '''
        plt.scatter(self.points[:,0], self.points[:,1], color= self.color)
    
class Line():
    def __init__(self, weight = 1, bias = 0, color = "black"):
        ##We can implicitly represent this line using a weight vector w and a 
        ##bias b. 
        ##The line then corresponds to the set of points x where: wTx+b=0.
        self.w = weight
        self.b = bias
        self.color = color
        #Calculate the weight vector.
        self.w_vector = [(-1-self.b)/self.w, 1]
        
    def plot_line(self, plot_range = [-1, 1]):
        '''Plot the line into the specified range. '''
        x_axis = np.linspace(plot_range[0], plot_range[1], 100)
        y_axis = self.w*x_axis + self.b
        plt.plot(x_axis, y_axis, color = self.color)
    
    def position_from_line(self, new_data):
        '''Relative position of a point to the line.
            
            args:
                new_data: array containing x and y coordinates.
            return:
                relative position of new_data to the line.
        '''
        #If the value is >, the point is above the line.
        return self.w_vector[0]*new_data[0]+ self.w_vector[1]*new_data[1] + self.b
    
    def category(self, category_1, category_2, new_data):
        '''Categorizes new_data as belonging to category_1 or category_2.
        
            args: 
                category_1: class Point.
                category_2: class Point.
                new_data: array containing x and y coordinates.
            
            return:
                String with the color of new_data.
        '''
        ##In order to test whether the point is blue or red, we just have to 
        ##check whether it is above or below the line.
        ##This can be achieved by checking the sign of wTx+b. 
        position_1 = self.position_from_line(category_1.center)
        position_new_data = self.position_from_line(new_data)
        if position_1*position_new_data >= 0:
            #In case both position_1.center and new_data are in the same side 
            #of the line, new_data belongs to category_1.
            return category_1.color
        else:
            return category_2.color
            
    
def example():
    '''Example to divide of data categorisation.
        
        example taken from: http://www.deepideas.net/deep-learning-from-scratch-ii-perceptrons/
    '''    
    ##Assume that we are given a dataset consisting of 100 points in the plane. 
    ##Half of the points are red and half of the points are blue.
    
    ##Create 50 red points centered at (-2,-2).
    red_data = Points("red", 50, [-2,-2])
    ##Create 50 blue points centered at (2,2).
    blue_data = Points('blue', 50, [2,2])
    ## Plot the red and blue points.
    red_data.plot_points()
    blue_data.plot_points()
    
    ##Apparently, we can draw a line y=−x that nicely separates the space into a red 
    ##region and a blue region:
    division_line = Line(-1,0)
    ##Plot a line y=-x.
    division_line.plot_line([-4,4])
    
    ##For example, if someone asks us what the color of the point (3,2) should be, 
    ##we’d best respond with blue. Even though this point was not part of the data 
    ##we have seen, we can infer this since it is located in the blue region of the 
    ##space.
    new_point = [3,2]
    print("The point, ", new_point, " is classified as:")
    print (division_line.category(red_data,blue_data, new_point))


