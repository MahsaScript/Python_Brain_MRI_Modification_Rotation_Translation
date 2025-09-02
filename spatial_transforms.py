# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 01:26:00 2021

@author: 
"""

# Import libraries
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X, Y, Z = np.mgrid[0:20:10j, 0:20:10j, 0:5:10j]  # Define X, Y, Z as data points


T = np.exp(-X**2 - Y**2 - Z**2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 3 Dimensions
scat = ax.scatter(X, Y, Z, c=Z.flatten(), alpha=0.5) # Scatter points
fig.colorbar(scat, shrink=0.5, aspect=5)
plt.savefig('3Dmesh.jpg')

from math import cos, sin
def rigid_transform(theta, omega, phi, X, Y, Z):
    if theta != 0: #rotate around X by theta angle      
       
        p = X # X`
        q =(Y*cos(theta) - Z*sin(theta)) # Y`  y cos θ − z sin θ
        r =(Y*sin(theta) + Z*cos(theta)) # Z`  y sin θ + z cos θ
        
        T = np.exp(-p**2 - q**2 - r**2)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # 3 Dimensions
        scat = ax.scatter(p, q, r, c=r.flatten(), alpha=0.5) # Scatter points
        fig.colorbar(scat, shrink=0.5, aspect=5)
        
    elif omega != 0: #rotate around Y by omega angle
        
        p = (X*cos(omega) + Z*sin(omega)) # X`  x cos θ + z sin θ 
        q = Y  # Y`
        r =((-X)*sin(omega) + Z*cos(omega)) # Z`  −x sin θ + z cos θ
        
        T = np.exp(-p**2 - q**2 - r**2)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # 3 Dimensions
        scat = ax.scatter(p, q, r, c=r.flatten(), alpha=0.5) # Scatter points
        fig.colorbar(scat, shrink=0.5, aspect=5)
        
        
    elif phi != 0:  #rotate around Z by phi angle
        #rotate around Z by phi angle
        p = (X*cos(phi) - Z*sin(phi)) # X`  x cos θ − y sin θ
        q = (X*sin(phi) + Y*cos(phi) ) # Y` x sin θ + y cos θ
        r = Z # Z`  
        T = np.exp(-p**2 - q**2 - r**2)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # 3 Dimensions
        scat = ax.scatter(p, q, r, c=r.flatten(), alpha=0.5) # Scatter points
        fig.colorbar(scat, shrink=0.5, aspect=5)

#rotate around X by theta angle  
rigid_transform(10,0,0,X, Y, Z)  

#rotate around Y by omega angle 
rigid_transform(0,10,0,X, Y, Z)  

#rotate around Z by phi angle 
rigid_transform(0,0,10,X, Y, Z)  

##Affine Transformation


def affine_transform_scaling(s, theta, omega, phi, X, Y, Z):
    if theta != 0: #rotate around X by theta angle      
       
        p = X # X`
        q =(Y*cos(theta) - Z*sin(theta)) # Y`  y cos θ − z sin θ
        r =(Y*sin(theta) + Z*cos(theta)) # Z`  y sin θ + z cos θ
        
        T = np.exp(-p**2 - q**2 - r**2)

        fig = plt.figure(figsize=(s, s))
        ax = fig.add_subplot(111, projection='3d')  # 3 Dimensions
        scat = ax.scatter(p, q, r, c=r.flatten(), alpha=0.5) # Scatter points
        fig.colorbar(scat, shrink=0.5, aspect=5)
        
    elif omega != 0: #rotate around Y by omega angle
        
        p = (X*cos(omega) + Z*sin(omega)) # X`  x cos θ + z sin θ 
        q = Y  # Y`
        r =((-X)*sin(omega) + Z*cos(omega)) # Z`  −x sin θ + z cos θ
        
        T = np.exp(-p**2 - q**2 - r**2)

        fig = plt.figure(figsize=(s, s))
        ax = fig.add_subplot(111, projection='3d')  # 3 Dimensions
        scat = ax.scatter(p, q, r, c=r.flatten(), alpha=0.5) # Scatter points
        fig.colorbar(scat, shrink=0.5, aspect=5)
        
        
    elif phi != 0:  #rotate around Z by phi angle
        #rotate around Z by phi angle
        p = (X*cos(phi) - Z*sin(phi)) # X`  x cos θ − y sin θ
        q = (X*sin(phi) + Y*cos(phi) ) # Y` x sin θ + y cos θ
        r = Z # Z`  
        T = np.exp(-p**2 - q**2 - r**2)

        fig = plt.figure(figsize=(s, s))
        ax = fig.add_subplot(111, projection='3d')  # 3 Dimensions
        scat = ax.scatter(p, q, r, c=r.flatten(), alpha=0.5) # Scatter points
        fig.colorbar(scat, shrink=0.5, aspect=5)

#rotate around X by theta angle  
affine_transform_scaling(10,10,0,0,X, Y, Z)  

#rotate around Y by omega angle 
affine_transform_scaling(10,0,10,0,X, Y, Z)  

#rotate around Z by phi angle 
affine_transform_scaling(10,0,0,10,X, Y, Z) 


import cv2
def get_type_transform_by_matrix():
    X, Y, Z = np.mgrid[0:20:10j, 0:20:10j, 0:5:10j]  # Define X, Y, Z as data points

    
    T = np.exp(-X**2 - Y**2 - Z**2)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 3 Dimensions
    scat = ax.scatter(X, Y, Z, c=Z.flatten(), alpha=0.5) # Scatter points
    src = cv2.imread('3Dmesh.jpg')
    rows, cols, ch = src.shape
      
    M1=[ [0.9045,  -0.3847, -0.1840,  10.0000],
        [ 0.2939,   0.8750, -0.3847,  10.0000],
        [ 0.3090,   0.2939,  0.9045,  10.0000],
        [ 0,        0,       0,        1.0000]]
    
  
    
    M2=[ [-0.0000,  -0.2598, 0.1500,  -3.0000],
          [0.0000,   -0.1500, -0.2598,  1.5000],
          [0.3000,   -0.0000,  0.0000,    0],
          [0,        0,        0,        1.0000]]
    
    
    M3=[ [0.7182,  -1.3727, -0.5660,  1.8115],
        [ -1.9236,  -4.6556, -2.5512, 0.2873],
        [ -0.6426,  -1.7985, -1.6285, 0.7404],
        [ 0,        0,       0,       1.0000]]
    
    #M1
    Xnew=(0.9045*X)+( -0.3847*Y)+( -0.1840*Z)+(10.0000*1)
    Ynew=( 0.2939*X)+(   0.8750*Y) +(-0.3847*Z)+( 10.0000*1)
    Znew= (0.3090*X)+(   0.2939*Y)+(  0.9045*Z)+(  10.0000*1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 3 Dimensions
    scat = ax.scatter(Xnew, Ynew, Znew, c=Znew.flatten(), alpha=0.5) # Scatter points
    fig.colorbar(scat, shrink=0.5, aspect=5)
    
    #M2
    Xnew=(-0.0000*X)+( -0.2598*Y)+(  0.1500*Z)+(-3.0000*1)
    Ynew=( 0.0000*X)+(   -0.1500*Y) +(-0.2598*Z)+( 1.5000*1)
    Znew= (0.3000*X)+(   -0.0000*Y)+(  0.0000*Z)+(  0*1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 3 Dimensions
    scat = ax.scatter(Xnew, Ynew, Znew, c=Znew.flatten(), alpha=0.5) # Scatter points
    fig.colorbar(scat, shrink=0.5, aspect=5)
    
    #M3
    Xnew=(0.7182*X)+( -1.3727*Y)+( -0.5660*Z)+(1.8115*1)
    Ynew=( -1.9236*X)+(   -4.6556*Y) +(-2.5512*Z)+( 0.2873*1)
    Znew= (-0.6426*X)+(  -1.7985*Y)+(  -1.6285*Z)+(  0.7404*1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 3 Dimensions
    scat = ax.scatter(Xnew, Ynew, Znew, c=Znew.flatten(), alpha=0.5) # Scatter points
    fig.colorbar(scat, shrink=0.5, aspect=5)
    
    
    
get_type_transform_by_matrix()