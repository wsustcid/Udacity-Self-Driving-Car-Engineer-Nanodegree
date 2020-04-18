'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-13 17:35:48
@LastEditTime: 2020-04-18 23:14:27
'''

""" === DESCRIPTION ===
The tools we have are 
  - color selection, 
  - region of interest selection, 
  - grayscaling, 
  - Gaussian smoothing, 
  - Canny Edge Detection 
  - and Hough Tranform line detection.  
1. Firstly, our goal is piece together a pipeline to detect the line segments in the image, the output should look something like 'examples/line-segments-example.jpg'.
2. Then average/extrapolate them and draw them onto the image for display.  
3. Finally, once we have a working pipeline, we will try it out on the video stream below. The output image should like "examples/laneLines_thirdPass.jpg".

Tips:
1. Some OpenCV functions that might be useful for this project are:
  - `cv2.inRange()` for color selection  
  - `cv2.fillPoly()` for regions selection  
  - `cv2.line()` to draw lines on an image given endpoints  
  - `cv2.addWeighted()` to coadd / overlay two images  
  - `cv2.cvtColor()` to grayscale or change color  
  - `cv2.imwrite()` to output images to file  
  - `cv2.bitwise_and()` to apply a mask to an image


2. Pipeline: read img -> covert to gray -> blur -> canny -> mask -> houghline -> fit lane lines -> draw line


Version 1.0.0:
1. 初始版本对虚线的拟合还不太稳定，老是跳变，还是canny边缘检测参数的问题;
2. 通过观察原始检测到的线段发现，跳变是由于检测到了横着的车道线 --> 剔除斜率为0的点； 以及道路边缘，导致有时右线跑到左边 --> 缩小ROI区域；最后效果还可以

"""

## Import packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os 

from helper_fun import grayscale, gaussian_blur, canny, region_of_interest
from helper_fun import hough_lines, fit_lane_lines, draw_lines, weighted_img

## TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.


def find_lane_lines(image):

  ## Covert to gray
  gray = grayscale(image)

  ## blur
  kernel_size = 5
  gray_blur = gaussian_blur(gray, kernel_size)

  ## Canny detection
  low_threshold = 50
  high_threshold = 150
  edges = canny(gray_blur, low_threshold, high_threshold)

  ## mask interest region
  polygon = np.array([[30, 540], [430, 330], [540, 330], [930, 540]])
  masked_edges = region_of_interest(edges, polygons=[polygon])

  ## Finding lines
  rho = 2
  theta = 2*np.pi/180
  threshold = 30 # minimum number line intersections in hough
  min_line_len = 20 # 这个参数过小，会导致线段过多，拟合车道线时斜率会不准确
  max_line_gap = 10
  lines = hough_lines(masked_edges, rho, theta, threshold, 
                        min_line_len, max_line_gap)

  ## draw lines on original image
  #img_lines = draw_lines(image, lines)
  
  ## draw lane lines
  lane_lines = fit_lane_lines(lines)
  lane_img = np.zeros_like(image, dtype=np.uint8)
  lane_img = draw_lines(lane_img, lane_lines, color=[255, 0, 0], thickness=6)

  img_lane = weighted_img(image, lane_img)

  return img_lane


if __name__ == "__main__":
  
  ## Read in an image
  img_name = os.listdir('./test_images/')[0]
  image = mpimg.imread(os.path.join('./test_images', img_name)) 
  print(type(image), 'dimensions:', image.shape) # (h:540,w:960,c:3)
  #plt.imshow(image) # if you wanted to show a single color channel image called 'gray', call as plt.imshow(gray, cmap='gray')
  #plt.show()
  
  ## Find Lane lines
  img_lane = find_lane_lines(image)
  
  ## save 
  mpimg.imsave(os.path.join('./test_images_output', 'lane_' + img_name), img_lane)
  #mpimg.imsave(os.path.join('./test_images_output', 'line_' + img_name), img_lines)

  plt.imshow(img_lane)
  plt.show()




