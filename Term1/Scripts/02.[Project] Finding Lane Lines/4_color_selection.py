'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-03-26 11:45:38
@LastEditTime: 2020-04-02 11:26:00
'''

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
image = mpimg.imread('lane.jpg')
print('This image is: ',type(image), 
         'with dimensions:', image.shape)
# image.shape = [height, width]

# Grab the x and y size and make a copy of the image
height = image.shape[0]
width = image.shape[1]


# Note: Always make a copy of arrays or other variables in Python. 
# If instead, you use "a = b" then all changes you make to "a" 
# will be reflected in "b" as well!
color_select = np.copy(image)

# Define color selection criteria
###### MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
red_threshold = 200
green_threshold = 200
blue_threshold = 200
######
# answer: 200 (recognize all 4 lane lines)
# If we set to 200, can extract two lines directly in front of the vehicle.

rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Do a boolean or with the "|" character to identify
# pixels below the thresholds
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
# thresholds is a 2-D boolean matrix, 
# The matrix elements are False only when the RGB values are all above the corresponding rgb_thresholds.
  
color_select[thresholds] = [0,0,0]

# Uncomment the following code if you are running the code locally and wish to save the image
mpimg.imsave("lane_color_selection.png", color_select)

# Display the image                 
plt.imshow(color_select)
plt.show()


