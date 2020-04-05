'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-03-26 11:45:37
@LastEditTime: 2020-04-02 21:36:52
'''

# Do all the relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in the image and convert to grayscale
image = mpimg.imread('edge.jpg')

gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
mpimg.imsave('edge_gray', gray, cmap='gray')

# Define a kernel size for Gaussian smoothing / blurring
kernel_size = 5 # Must be an odd number (3, 5, 7...)
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
mpimg.imsave('edge_gray_blur', blur_gray, cmap='gray')

# Define our parameters for Canny and run it
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
mpimg.imsave('edge_canny', edges, cmap='gray')

# Display the image
plt.imshow(edges, cmap='Greys_r')
#plt.show()