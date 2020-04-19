'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.1.0
@Date: 2020-04-13 17:35:48
@LastEditTime: 2020-04-19 12:01:43
'''

""" === DESCRIPTION ===
1. Firstly, our goal is piece together a pipeline to detect the line segments in the image, the output should look something like 'examples/line-segments-example.jpg'.
 The tools we have are 
  - color selection, 
  - region of interest selection, 
  - grayscaling, 
  - Gaussian smoothing, 
  - Canny Edge Detection 
  - and Hough Tranform line detection.  
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


==== CHANGE LOG ====
Version 1.0.0:
1. 针对虚线车道线检测老是跳变的问题：通过观察原始检测到的线段发现，跳变是由于检测到了横着的车道线以及道路边缘；解决方案为：设置斜率区间，剔除横线 + 缩小ROI区域；
2. 问题：当前版本起始点是每次检测到的车道线起始点，虚线有时并不是图片最下方，导致视觉不连续；

Version 1.1.0
1. 参考项目：https://github.com/woges/UDACITY-self-driving-car， 本项目主要创新如下：
  - 先通过color selection选出黄色车道线图和白色车道线图；然后两张图同步执行pipeline；最后addweightted合并检测到的两张车道线图(权重均为1)
  - 设置更为精确地斜率区间： if slope_calc < -0.55 and slope_calc > -0.85
                         elif slope_calc > 0.49 and slope_calc < 0.7

  - 使用比例设置ROI,避免视频分辨率不同导致的多次调参：
    imshape = image_2.shape
    vertices = np.array([[(0.1*imshape[1],imshape[0]),
                          (0.48*imshape[1], imshape[0]/1.7),
                          (0.52*imshape[1], imshape[0]/1.7), 
                          (0.95*imshape[1],imshape[0])]], dtype=np.int32)
  - canny 参数:
    rho = 1 
    theta = np.pi/150
    threshold = 8
    min_line_length = 12 
    max_line_gap = 10 

2. 不再检测起点终点，通过已拟合直线，根据预定义起点终点y，推断对应x
3. 问题：但对于challenge.mp4, 可能由于ROI及canny参数问题，导致中途会没有点检测到，导致车道线拟合失败；或者也由于canny检测对环境变化的鲁棒性不够（遮挡，阴影等，后期可尝试参考项目中的方法）  --> 问题的关键就是 特征点检测 与 直线拟合
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
  imshape = image.shape
  polygon = np.array([[(0.1*imshape[1],imshape[0]),
                       (0.48*imshape[1], imshape[0]/1.7),
                       (0.52*imshape[1], imshape[0]/1.7), 
                       (0.95*imshape[1],imshape[0])]], dtype=np.int32)
  masked_edges = region_of_interest(edges, polygons=[polygon])

  ## Finding lines
  rho = 1
  theta = 1*np.pi/180
  threshold = 8 # minimum number line intersections in hough
  min_line_len = 12 # 这个参数过小，会导致线段过多，拟合车道线时斜率会不准确
  max_line_gap = 10
  lines = hough_lines(masked_edges, rho, theta, threshold, 
                        min_line_len, max_line_gap)

  ## draw lines on original image
  #img_lines = draw_lines(image, lines)
  
  ## draw lane lines
  lane_lines = fit_lane_lines(lines, imshape=imshape)
  lane_img = np.zeros_like(image, dtype=np.uint8)
  lane_img = draw_lines(lane_img, lane_lines, color=[255, 0, 0], thickness=6)

  img_lane = weighted_img(image, lane_img)

  return img_lane


if __name__ == "__main__":
  
  ## Read in an image
  img_name = os.listdir('./test_images/')[5]
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




