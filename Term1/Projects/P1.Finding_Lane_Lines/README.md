# Finding Lane Lines on the Road 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

## Overview

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project we will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  




The Project
---

In this project, we will use the tools we learned about in the Term1-lesson2 to identify lane lines on the road.  

1. We will develop a pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip `./examples/raw-lines-example.mp4` (also contained in this repository) to see what the output should look like.

2. Next we will average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result in the video `./exampels/P1_example.mp4`.  
3. Ultimately, we would like to draw just one line for the left side of the lane, and one for the right.



TODO: Creating a Great Writeup
---

For this project, a great writeup should provide a detailed response to the "Reflection" section of the [project rubric](https://review.udacity.com/#!/rubrics/322/view). There are three parts to the reflection:

1. Describe the pipeline

2. Identify any shortcomings

3. Suggest possible improvements

We encourage using images in your writeup to demonstrate how your pipeline works.  

All that said, please be concise!  We're not looking for you to write a book here: just a brief description.

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup. Here is a link to a [writeup template file](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md). 

