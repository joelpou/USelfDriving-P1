# **Finding Lane Lines on the Road**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

#[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale. Using this grayscale copy I applied a Gaussian smoothing filter
since this would enable me to remove unwanted detail and noise. I used a kernel_size of 5 which gave me good results. After pre-processing the image the second step was to use Canny edge detection to get all possible lines on image. I used a value of 50 and 150 for low and high thresholds. Afterwards, An appropriate polygon was determined after a bit of trial and error and this region was masked out by using a region_of_interest function. The fourth step was to use hough_lines function to detect lines on masked_edges image and also trigger draw_lines function in order to draw new lines based on the position of the Hough lines. Finally, the last step was to improved the drawn lines by means of the weighted_img function.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by initializing four empty lists. This was done to separate x and y points into two different groups based on the slope of the Hough line detected. These lists were then passed to get_line_info function and by using numpy's linalg.lstsq method I could return a fitted line that passes through all points, thus giving me the slope and intercept of this line. After getting this line I extrapolate the initial and final x points by using the standard line equation and slope-to-point formula in order to draw the new line.


If you'd like to include images to show how the pipeline works, here is how to include an image:

#![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline

For some reason I'm getting an image looking like this when I apply Gaussian smoothing:

[image1]: ./issues/gaussianoutput.png "blur_gray"

Also, Canny output is outputting image in purple, not in black and white. I suspect this has to do with the right image formatting using the imshow and show functions to display images?

[image1]: ./issues/cannyoutput.png "canny"

Last but not least, getting issues with drawing lines using least square methods approach. Sometimes lines drift away and it usually happens when the length of a point is zero (second image). In this case I just ignore this line but I get a gap in the video output in this frame. Also, for some reason I haven't figured out, lines are getting inverted when I store them in lists base on the slope.

[image1]: ./issues/Selection_002.png "lstsq1"
[image1]: ./issues/Selection_003.png "lstsq2"


### 3. Suggest possible improvements to your pipeline

I need to improve my draw_lines function to better approximate lines that are ignored out.
