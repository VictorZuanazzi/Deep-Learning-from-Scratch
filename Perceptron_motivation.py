# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 18:28:26 2018

@author: victzuan
Source: http://www.deepideas.net/deep-learning-from-scratch-ii-perceptrons/

"""
import numpy as np
import matplotlib.pyplot as plt

#Assume that we are given a dataset consisting of 100 points in the plane. 
#Half of the points are red and half of the points are blue.

#Create 50 red points centered at (-2,-2).
red_points = np.random.randn(50,2) -2*np.ones((50,2))

#Create 50 blue points centered at (2,2).
blue_points = np.random.randn(50,2) + 2*np.ones((50,2))

# Plot the red and blue points.
plt.scatter(red_points[:,0], red_points[:,1], color = "red")
plt.scatter(blue_points[:,0], blue_points[:,1], color = "blue")

#For example, if someone asks us what the color of the point (3,2) should be, 
#we’d best respond with blue. Even though this point was not part of the data 
#we have seen, we can infer this since it is located in the blue region of the 
#space.

#Apparently, we can draw a line y=−x that nicely separates the space into a red 
#region and a blue region:

#Plot a line y=-x.
x_axis = np.linspace(-4,4,100)
y_axis = -x_axis
plt.plot(x_axis, y_axis, color ='black')

#We can implicitly represent this line using a weight vector w and a bias b. 
#The line then corresponds to the set of points x where: wTx+b=0.

#In the case above, we have w=(1,1)T and b=0. Now, in order to test whether the 
#point is blue or red, we just have to check whether it is above or below the 
#line. This can be achieved by checking the sign of wTx+b. 
#If it is positive, then x is above the line. 
#If it is negative, then x is below the line.
 
#Let’s perform this test for our example point (3,2)T:

#(1, 1)⋅(3, 2)=5
#Since 5 > 0, we know that the point is above the line and, therefore, should 
#be classified as blue.