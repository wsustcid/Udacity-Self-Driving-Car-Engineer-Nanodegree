'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-03-26 11:45:38
@LastEditTime: 2020-04-02 11:27:07
'''

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
image = mpimg.imread('lane.jpg')
print("image shape", image.shape)

# Grab the x and y size and make a copy of the image
height = image.shape[0]
width  = image.shape[1]

color_select = np.copy(image)
line_image   = np.copy(image)

# Define color selection criteria
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Define the vertices of a triangular mask.
# The standard location of the origin (x=0, y=0) for images is in the top left corner 
# with y values increasing downward and x increasing to the right. 
# Think about an image as a matrix. 

# MODIFY THESE VALUES TO ISOLATE THE REGION 
# WHERE THE LANE LINES ARE IN THE IMAGE
left_bottom = [0, 539]
right_bottom = [959, 539]
apex = [480, 330]

# Perform a linear fit (y=Ax+B) to each of the three sides of the triangle
# np.polyfit returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

## May be better with a trapezoid ##

# Mask pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                    (image[:,:,1] < rgb_threshold[1]) | \
                    (image[:,:,2] < rgb_threshold[2])

# Find the region inside the lines
# np.meshgrid(): given one-dimensional coordinate arrays x1, x2,…, xn.
# Return coordinate matrices from coordinate vectors.
# e.g., XX = [[0,1,...,959],...,[0,1,...,959]]
# YY = [[0,...,0],...,[539,...,539]]
XX, YY = np.meshgrid(np.arange(0, width), np.arange(0, height))

region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))
                    
# Mask color and region selection
color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]

# Color pixels red where both color and region selections met
line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

# save the results
mpimg.imsave("lane_color_selection_region_mask.png", color_select)
mpimg.imsave("lane_line_identified.png", line_image)


# Display the final image and show region and color selections
plt.imshow(line_image)
x = [left_bottom[0], right_bottom[0], apex[0], left_bottom[0]]
y = [left_bottom[1], right_bottom[1], apex[1], left_bottom[1]]
plt.plot(x, y, lw=3)
plt.savefig('lane_line_identified_region_show.png')

plt.show()