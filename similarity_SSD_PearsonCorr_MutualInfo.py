# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 06:08:21 2021

@author: 
"""
import pandas as pd
import numpy as np
from PIL import Image

img = Image.open('BrainMRI_1.jpg')

img2 = Image.open('BrainMRI_2.jpg')

# images of size n x p
n=256 #width
p=256 #height
Nimg = img.resize((n,p))   # image resizing 220*180

Nimg2 = img2.resize((n,p)) # image resizing
Nimg.save('BrainMRI_11.jpg')
Nimg2.save('BrainMRI_22.jpg')



import cv2
image = cv2.imread('BrainMRI_11.jpg')  # image reading first image 
image2 = cv2.imread('BrainMRI_22.jpg') # image reading second image

gray_image  = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY) # Converting to gray first image
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) # Converting to gray second image

histogram1 = cv2.calcHist([gray_image], [0], 
                              None, [256], [0, 256])

histogram2 = cv2.calcHist([gray_image2], [0], 
                              None, [256], [0, 256])
data1=np.array(histogram1)
data2=np.array(histogram2)
data=np.concatenate((data1, data2), axis=1)

def sum_square_difference(x,y):
    ssd=sum((x - y)**2)
    return ssd
   

print("sum_square_difference %d" %(sum_square_difference(data1, data2)))


print("Pearson’s Correlation")
print("Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))")
def pearson_r(x, y):
  # Assume len(x) == len(y)
  n = len(x)
  sum_x = float(sum(x))
  sum_y = float(sum(y))
  sum_x_sq = sum(xi*xi for xi in x)
  # sum_x_sq = np.prod(x)
  sum_y_sq = sum(yi*yi for yi in y)
  # sum_y_sq = np.prod(y)
  psum = sum(xi*yi for xi, yi in zip(x, y))
  # psum=np.prod([x,y], axis=1)
  num = psum - (sum_x * sum_y/n)
  den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
  if den == 0: return 0
  return num / den
print("Pearson’s Correlation:")
print(pearson_r(data1, data2))

import sys
def shannon_entropy_function(A, mode="auto", verbose=False):

    A = np.asarray(A)

    # Determine distribution type
    if mode == "auto":
        condition = np.all(A.astype(float) == A.astype(int))
        if condition:
            mode = "discrete"
        else:
            mode = "continuous"
    if verbose:
        print(mode, file=sys.stderr)
    # Compute shannon entropy
    pA = A / A.sum()
    # Remove zeros
    pA = pA[np.nonzero(pA)[0]]
    if mode == "continuous":
        return -np.sum(pA*np.log2(A))  
    if mode == "discrete":
        return -np.sum(pA*np.log2(pA))   

def mutual_information_images(x,y, mode="auto", normalized=False):
    """
    I(X, Y) = H(X) + H(Y) - H(X,Y)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # Determine distribution type
    if mode == "auto":
        condition_1 = np.all(x.astype(float) == x.astype(int))
        condition_2 = np.all(y.astype(float) == y.astype(int))
        if all([condition_1, condition_2]):
            mode = "discrete"
        else:
            mode = "continuous"

    H_x = shannon_entropy_function(x, mode=mode)
    H_y = shannon_entropy_function(y, mode=mode)
    H_xy = shannon_entropy_function(np.concatenate([x,y]), mode=mode)

    # Mutual Information images 
    I_xy = H_x + H_y - H_xy
    if normalized:
        return I_xy/np.sqrt(H_x*H_y)
    else:
        return  I_xy
    
mutual_information_result = mutual_information_images(data1, data2)
print("mutual_information_images %d" %(mutual_information_result))