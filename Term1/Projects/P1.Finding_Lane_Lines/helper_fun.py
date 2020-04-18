'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-13 21:25:20
@LastEditTime: 2020-04-18 22:17:55
'''

import numpy as np
import cv2

from scipy import optimize

def grayscale(img):
    """Applies the Grayscale transform
    Input: an image with RGB channels 
    Return: an image with only one color channel
    Note: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""

    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""

    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, polygons):
    """ Applies an image mask.
    Inputs:
    img:
    polygons: Array of polygons where each polygon is represented as an array of points. e.g. [polygon1, polygon2] and polygon1=[[x1, y1],...]
    
    Return: an image that keeps the region defined by the polygon
    formed from `vertices`, the rest of the image is set to black.
    """
    
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        fill_color = (255,) * channel_count
    else:
        fill_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, polygons, fill_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """ 
    Inputs:
      - img: the output of a Canny transform.
      - rho: distance resolution in pixels of the Hough grid
      - theta: angular resolution in radians of the Hough grid
      - threshold: minimum number of intersections in Hough grid cell
      - min_line_length: minimum number of pixels making up a line
      - max_line_gap: maximum gap in pixels between connectable line segments (用来连线)

    Return: 3D array which contains start points and end points of the lines. e.g. [[[x1, y1, x2,y2]], [[xx,xx,xx,xx]],...,]
    """

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    return lines

def fit_lane_lines(lines):
    """ Fitting two lane lines (left and right) from multiple line segments
    Input: A 3D array that contains multiple lines 
    Output: A 3D array that contains only two lines

    Hit:
    #((y2-y1)/(x2-x1)) decides which segments are part of the left
    # line vs. the right line. 
    # average the position of each of the lines 
    # and extrapolate to the top and bottom of the lane.
   
    """
        
    left_points_x, left_points_y, left_slope = [], [], []
    right_points_x, right_points_y, right_slope = [], [], []
    # note here the shape of the line is (1,4), e.g. [[1,2,3,4]]
    # one dim array is not iterable!

    delta = 0.5 # to delete the horizontal lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2-y1)/(x2-x1)
            if slope < -delta:
                left_slope.append(slope)
                left_points_x.append([x1,x2])
                left_points_y.append([y1,y2])
            elif slope > delta:
                right_slope.append(slope)
                right_points_x.append([x1,x2])
                right_points_y.append([y1,y2])
    
    # fit a line 
    left_k = np.mean(left_slope)
    left_x0 = np.mean(left_points_x)
    left_y0 = np.mean(left_points_y)
    left_b = left_y0 - left_k * left_x0
    
    # infer start and end points by y
    left_y1 = np.max(left_points_y)
    left_y2 = np.min(left_points_y)
    left_x1 = (left_y1 - left_b)/left_k
    left_x2 = (left_y2 - left_b)/left_k

    # fit a line 
    right_k = np.mean(right_slope)
    right_x0 = np.mean(right_points_x)
    right_y0 = np.mean(right_points_y)
    right_b = right_y0 - right_k * right_x0
    
    # infer start and end points by y
    right_y1 = np.max(right_points_y)
    right_y2 = np.min(right_points_y)
    right_x1 = (right_y1 - right_b)/right_k
    right_x2 = (right_y2 - right_b)/right_k

    lane_lines = np.array([[[left_x1,left_y1,left_x2,left_y2]],
                           [[right_x1,right_y1,right_x2,right_y2]]], dtype=np.uint32)
    return lane_lines
    

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """ Draw `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    return img


def weighted_img(initial_img, img, alpha=0.8, beta=1., gamma=0.):
    """
    Params:
      - img: the output of the hough_lines(), An image with lines drawn on it. Should be a blank image (all black) with lines drawn on it
      - initial_img: the image before any processing.
    Return: a result image computed by initial_img * α + img * β + γ
    
    Note: initial_img and img must be the same shape!
    """
    
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)
    

def straight_line(x, A, B):
    return A*x + B

def quadratic_curve(x, A, B, C):
    return A*x*x + B*x + C

def cubic_curve(x, A, B, C, D):
  return A*x*x*x + B*x*x + C*x + D

def fit_curve_lane_lines(lines):
    """ Fitting two curve lane lines (left and right) from multiple line segments
    Input: A 3D array that contains multiple lines 
    Output: pixel indexes that draw the lane curve

    Hit: 
    1. scipy的optimize包中提供了一个专门用于曲线拟合的函数curve_fit()

    """
        
    left_points_x, left_points_y = [], []
    right_points_x, right_points_y = [], []
    # note here the shape of the line is (1,4), e.g. [[1,2,3,4]]
    # one dim array is not iterable!

    delta = 0.1 # to delete the horizontal lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2-y1)/(x2-x1)
            if slope < -delta:
                left_points_x.append(x1)
                left_points_x.append(x2)
                left_points_y.append(y1)
                left_points_y.append(y2)
            elif slope > delta:
                right_points_x.append(x1)
                right_points_x.append(x2)
                right_points_y.append(y1)
                right_points_y.append(y2)
    
    # fit a left curve lane
    # points in curve_fit should be 1-D array 
    A1, B1, C1 = optimize.curve_fit(quadratic_curve, 
                                    left_points_x, 
                                    left_points_y)[0]

    # infer left line start and end points by x
    left_x1 = np.min(left_points_x)
    left_x2 = np.max(left_points_x)
    
    # calculate left curve pixel indexes
    left_x = np.arange(left_x1, left_x2, 1)
    left_y = np.array(quadratic_curve(left_x, A1, B1, C1), dtype=np.int64)
    

    # fit a right curve lane
    A2, B2, C2 = optimize.curve_fit(quadratic_curve, 
                                    right_points_x,
                                    right_points_y)[0]
    
    # infer right line start and end points by x
    right_x1 = np.max(right_points_x)
    right_x2 = np.min(right_points_x)

    # calculate right curve pixel indexes
    right_x = np.arange(right_x2, right_x1, 1)
    right_y = np.array(quadratic_curve(right_x, A2, B2, C2), dtype=np.int64)

    curve_lines = [left_x, left_y, right_x, right_y]
    
    return curve_lines
    
def draw_curve_lines(img, curve_lines, color=[255,0,0], thickness=-1, size=5):
    """ Draw curve lines by multiple rectangles.
    cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])     
    """
    
    #color = (255,0,0)
    # left curve
    for i in range(len(curve_lines[0])):
        x, y = curve_lines[0][i], curve_lines[1][i]
        up_left = (x-size, y-size)
        down_right = (x+size, y+size)

        cv2.rectangle(img, up_left, down_right, color, thickness)
    
    # right curve
    for i in range(len(curve_lines[2])):
        x, y = curve_lines[2][i], curve_lines[3][i]
        up_left = (x-size, y-size)
        down_right = (x+size, y+size)

        cv2.rectangle(img, up_left, down_right, color, thickness)
    
    return img
