'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-14 16:09:27
@LastEditTime: 2020-04-14 17:37:14
'''

""" Drawing lanes over video!
We test our solution on two provided videos:
  - solidWhiteRight.mp4
  - solidYellowLeft.mp4

Tips:
1. To speed up the testing process, we can use a shorter subclip
   - add .subclip(start_second,end_second) to the end of the VideoFileClip
   - clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)

"""

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from lane_image import find_lane_lines

## display video in ipython notebook
#from IPython.display import HTML
#HTML("""
#<video width="960" height="540" controls>
#  <source src="{0}">
#</video>
#""".format(white_output))


## Load video
clip = VideoFileClip("test_videos/solidWhiteRight.mp4")

## Process
#Note: this function expects color images!!
process_clip = clip.fl_image(find_lane_lines) 

## save to file
output = 'test_videos_output/solidWhiteRight.mp4'
process_clip.write_videofile(output, audio=False)