'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-14 17:17:45
@LastEditTime: 2020-04-19 12:03:18
'''

""" Try to fit a curve lane line
v1.0.0:
待解决： 解决中途检测不到特征点的问题！！！
"""


## Import packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os 

from helper_fun import grayscale, gaussian_blur, canny, region_of_interest
from helper_fun import hough_lines, fit_lane_lines, draw_lines, weighted_img
from helper_fun import fit_curve_lane_lines, draw_curve_lines


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
  polygon = np.array([[(0.1*imshape[1], imshape[0]),
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
  lane_lines = fit_lane_lines(lines)
  lane_img = np.zeros_like(image, dtype=np.uint8)
  lane_img = draw_lines(lane_img, lane_lines, color=[255, 0, 0], thickness=6)

  img_lane = weighted_img(image, lane_img)

  return img_lane


def find_curve_lane_lines(image):

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
  #polygon = np.array([[150, 720], [500, 500], [900, 500], [1100, 720]])
  imshape = image.shape
  polygon = np.array([[(0.1*imshape[1],imshape[0]),(0.48*imshape[1], imshape[0]/1.7), \
            (0.52*imshape[1], imshape[0]/1.7), (0.95*imshape[1],imshape[0])]], dtype=np.int32)
  #polygon = np.array([[30, 540], [430, 330], [540, 330], [930, 540]])
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
  lane_lines = fit_curve_lane_lines(lines)
  lane_img = np.zeros_like(image, dtype=np.uint8)
  lane_img = draw_curve_lines(lane_img, lane_lines)

  img_lane = weighted_img(image, lane_img)

  return img_lane


if __name__ == "__main__":
  from moviepy.editor import VideoFileClip


  ## Load video
  clip = VideoFileClip("test_videos/challenge.mp4")

  ## Process
  #Note: this function expects color images!!
  process_clip = clip.fl_image(find_lane_lines) 

  ## save to file
  output = 'test_videos_output/challenge.mp4'
  process_clip.write_videofile(output, audio=False)




