# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 05:10:47 2021

@author: 
"""
# Import libraries
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import csv

def interpolation_functions():
    X, Y, Z = np.mgrid[0:20:10j, 0:20:10j, 0:5:10j]  # Define X, Y, Z as data points
    
    T = np.exp(-X**2 - Y**2 - Z**2)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 3 Dimensions
    scat = ax.scatter(X, Y, Z, c=Z.flatten(), alpha=0.5) # Scatter points
    p=0.1
    q=0.2
    t=(p,q)  
    M1=[ [1,  0, p],
        [ 0,  1, q],
        [ 0,  0, 1]]
       
    #M1
    Xnew=(1*X)+(0*Y)+( p*Z)
    Ynew=(0*X)+(1*Y) +(q*Z)
    Znew=(0*X)+(0*Y)+(1*Z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 3 Dimensions
    scat = ax.scatter(Xnew, Ynew, Znew, c=Znew.flatten(), alpha=0.5) # Scatter points
    fig.colorbar(scat, shrink=0.5, aspect=5)

print("Assignment 4 - part a: Plot different translation")
interpolation_functions()

def sum_square_difference(x,y):
    ssd=np.sum((x - y)**2)
    return ssd

def two_dim_registration_saving_SSD():
    # Open the image files.
    img1_color = cv2.imread("BrainMRI_3.jpg")  # Image to be aligned.
    img2_color = cv2.imread("BrainMRI_1.jpg")    # Reference image.
      
    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape
      
    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)
      
    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)
      
    # Match features between the two images.
    # We create a Brute Force matcher with 
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
      
    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)
      
    # Sort matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)
      
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*90)]
    no_of_matches = len(matches)
      
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    ssd_1_2=[]  
    with open('SSD_BrainMRI_1_3.csv', 'w', encoding='UTF8') as f:
        csvcreator = csv.writer(f, delimiter =',')
       
        for i in range(len(matches)):
          p1[i, :] = kp1[matches[i].queryIdx].pt
          p2[i, :] = kp2[matches[i].trainIdx].pt
          ssd_1_2.append(sum_square_difference(p1[i, :], p2[i, :]))
          
          
        csvcreator.writerow(ssd_1_2)
      
    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
      
    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color,
                        homography, (width, height))
      
    # Save the output.
    cv2.imwrite('registration_BrainMRI_1_and_3.jpg', transformed_img)
    image = cv2.imread('registration_BrainMRI_1_and_3.jpg')
    plt.figure()
    plt.imshow(image) 
    plt.show()  # display image

print("Assignment 4 - part b: Save SSD on csv file")
two_dim_registration_saving_SSD()


def rotation_function_by_angle():
    image = cv2.imread("BrainMRI_1.jpg")    # Reference image.
    row,col = image.shape[0:2]
    center=tuple(np.array([col,row])/2)
    rot_mat = cv2.getRotationMatrix2D(center,30,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    cv2.imwrite('rotate_30_degree.jpg', new_image)
    image = cv2.imread('rotate_30_degree.jpg')
    plt.figure()
    plt.imshow(image) 
    plt.show()  # display image
    
print("Assignment 4 - part c: Rotate image by the angle (For instance: 30 Degree)")
rotation_function_by_angle()


import math
import matplotlib.pyplot as plt

def two_dim_registration__just_rotations():
    # Open the image files.
    img1_color = cv2.imread("rotate_30_degree.jpg")  # Image to be aligned.
    img2_color = cv2.imread("BrainMRI_1.jpg")    # Reference image.
      
    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape
      
    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)
      
    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)
      
    # Match features between the two images.
    # We create a Brute Force matcher with 
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
      
    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)
      
    # Sort matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)
      
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*90)]
    no_of_matches = len(matches)
      
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    ssd_just_rotation=[]  
    with open('SSD_BrainMRI_1_Align_Just_For_Rotation.csv', 'w', encoding='UTF8') as f:
        csvcreator = csv.writer(f, delimiter =',')
       
        for i in range(len(matches)):
          p1[i, :] = kp1[matches[i].queryIdx].pt
          p2[i, :] = kp2[matches[i].trainIdx].pt
          # ----SSD = math.hypot----
          ssd_just_rotation.append(sum_square_difference(p1[i, :],p2[i, :]))

        csvcreator.writerow(ssd_just_rotation)
     
        plt.plot(ssd_just_rotation)
        plt.xlabel('iteration')
        plt.ylabel('SSD')
        plt.show()
    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
      
    # Use this matrix to transform the
    # colored image wrt the reference image.
    aligned_rotated_image = cv2.warpPerspective(img1_color,
                        homography, (width, height))
      
    # Save the output.
    cv2.imwrite('registration_BrainMRI_1_Just_For_Rotation.jpg', aligned_rotated_image)
    image = cv2.imread('registration_BrainMRI_1_Just_For_Rotation.jpg')
    plt.figure()
    plt.imshow(image) 
    plt.show()  # display image
 

print("Assignment 4 - part d: 2d registration minimizing SSD and considering only rotations.")
two_dim_registration__just_rotations()

def gradient_descent_vs_adam_for_minimizing_SSD():
        # Open the image files.
    img1_color = cv2.imread("BrainMRI_2.jpg")  # Image to be aligned.
    img2_color = cv2.imread("BrainMRI_1.jpg")    # Reference image.
      
    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape
      
    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)
      
    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)
      
    # Match features between the two images.
    # We create a Brute Force matcher with 
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
      
    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)
      
    # Sort matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)
      
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*90)]
    no_of_matches = len(matches)
      
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    ssd_gradient_descent=[]       
       
    #------------ssd_gradient_descent------------
    cur_x =900 # The algorithm starts at x=900
    rate = 0.5 # Learning rate
    iters = 0 #iteration counter
    prev_x = cur_x #Store current x value in prev_x
    for i in range(len(matches)):
      p1[i, :] = kp1[matches[i].queryIdx].pt
      p2[i, :] = kp2[matches[i].trainIdx].pt
      # ----SSD ----
      ssd=sum_square_difference(p1[i, :],p2[i, :])
      cur_x =abs( cur_x - rate * ssd)
      iters = iters+1 #iteration count
    print("Iteration for gradient descent",iters,"\nX optimal value is",cur_x) 
    
    print("The local minimum with gradient descent happens at", cur_x)
    
    #------------ssd_Adam_Optimizer-----------
    alpha = 0.01
    beta_1 = 0.9
    beta_2 = 0.999						#initialize the values of the parameters
    epsilon = 1e-8
    theta_0 = 400	#theta_0 is optimal SSD - initialize the ssd
    m_t = 0 
    v_t = 0 
    t = 0

    for i in range(len(matches)): #till it gets converged
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt
        # ----SSD ----
        ssd=sum_square_difference(p1[i, :],p2[i, :])
        t+=1
        g_t = ssd		#computes the gradient of the stochastic function
        m_t = beta_1*m_t + (1-beta_1)*g_t	#updates the moving averages of the gradient
        v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)	#updates the moving averages of the squared gradient
        m_cap = m_t/(1-(beta_1**t))		#calculates the bias-corrected estimates
        v_cap = v_t/(1-(beta_2**t))		#calculates the bias-corrected estimates
        theta_0_prev = theta_0
        theta_0 = theta_0 - (alpha*m_cap)/(math.sqrt(v_cap)+epsilon)	#updates the parameters

    print("Iteration for Adam Optimizer",len(matches),"\nX optimal value is",theta_0) 
    
    print("The local minimum with Adam Optimizer happens at", theta_0)

    
    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
      
    # Use this matrix to transform the
    # colored image wrt the reference image.
    aligned_rotated_image = cv2.warpPerspective(img1_color,
                        homography, (width, height))
      
    # Save the output.
    cv2.imwrite('optimal_ssd_BrainMRI_2.jpg', aligned_rotated_image)
    image = cv2.imread('optimal_ssd_BrainMRI_2.jpg')
    plt.figure()
    plt.imshow(image) 
    plt.show()  # display image
    #------------------------


print("Assignment 4 - part e:  Gradient descent for minimizing SSD, considering both translation and rotation.")
gradient_descent_vs_adam_for_minimizing_SSD()




