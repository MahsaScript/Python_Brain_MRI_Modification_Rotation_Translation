# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 05:12:57 2021

@author: mahsa
"""

from PIL import Image

img = Image.open('a.jpg')

img2 = Image.open('b.jpg')

# images of size n x p
n=480 #width
p=640 #height
Nimg = img.resize((n,p))   # image resizing 480*640

Nimg2 = img2.resize((n,p)) # image resizing
Nimg.save('aa.jpg')
Nimg2.save('bb.jpg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.figure()
plt.imshow(Nimg) 
plt.show()  # display image

plt.figure()
plt.imshow(Nimg2) 
plt.show()  # display image

import cv2
image = cv2.imread('aa.jpg')  # image reading first image 
image2 = cv2.imread('bb.jpg') # image reading second image

gray_image  = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY) # Converting to gray first image
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) # Converting to gray second image

histogram1 = cv2.calcHist([gray_image], [0], 
                              None, [256], [0, 256])

histogram2 = cv2.calcHist([gray_image2], [0], 
                              None, [256], [0, 256])

print("Part 1: Joint histogram - section a")
print(" calculates the joint histogram of two images of the same size")
from matplotlib import pyplot as plt

img = cv2.imread('aa.jpg',0)
plt.hist(img.ravel(),256,[0,256]); plt.show()


img2 = cv2.imread('bb.jpg',0)
plt.hist(img2.ravel(),256,[0,256]); plt.show()

sigma_hist_1 = sum(histogram1)
sigma_hist_2 = sum(histogram2)

print("Part 1: Assertion Sigma Histogram = n*p - section b")
print("Prove that images in size 220*180 Sigma Histogram,j(i,j)=n*p")
# Prove that images in size 220*180 Sigma Hi,j(i,j)=n*p
print("Sigma of Fist Histogram= %d & n*p= %d" %(sigma_hist_1, n*p))
print("Sigma of Second Histogram= %d & n*p= %d" %(sigma_hist_2, n*p))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
print("Part 1: Logarithmic Scale - section c")
print("Visualize joint hist by using the logarithmic scale")
data1=np.array(histogram1)
data2=np.array(histogram2)
data=np.concatenate((data1, data2), axis=1)

# plt.hist2d(data1,data2,bins,norm=LogNorm())
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# some random data
x = np.random.normal(size=100000)
y = x * 3.5 + np.random.normal(size=100000)

histogram1=np.squeeze(histogram1)
histogram2=np.squeeze(histogram2)

data1=np.squeeze(data1)
data2=np.squeeze(data2)


ax1 = sns.jointplot(x=data1, y=data2)
ax1.ax_joint.cla()
plt.sca(ax1.ax_joint)

plt.hist2d(data1, data2, bins=(40, 40), cmap=cm.jet);

