# Term 1

## 1. Welcome

### 1.1 Overview of ND Program

There are two approaches of **autonomous development**:

- **A robotic approach:** which fuses output from a suit of sensors to directly measure the vehicle surrounding, and the navigate accordingly.
- **A deep learning approach:** which allows self-driving car to learn how to drive by mimicking human driver behaviors.

***In this nanodegree program, you will be working with both approaches, and finally, you will control a real self-driving car!***

### 1.2 What Projects will you build

- Finding lane lines
- Behavior cloning
- Advanced lane finding and vehicle detection
- Sensor fusion
- Localization
- Controllers
- Path planning
- Put your code to a real self-driving car



## 2. [Project] Finding Lane Lines

### 2.1 Setting up problem

*When we drive we use our eyes to figure out **how fast to go** and **where  the lane lines are** and **where to turn**.*

Our goal in this module is to write code to **identify and track** the position of the lane lines **in a series of image**.

#### **Quiz?**

Which of the following features could be useful in the identification of lane lines on the road?

- Color
- Shape
- Orientation
- Position in the image

### 2.2 Color selection

#### **Finding the lane lines using color.**

1. Selecting the white pixels in an image. (Since the lane lines are white)
   - Color in digital images means that our image is actually made up of a stack of three images, one each for **red, green, and blue**. 
   - Color channels contain pixels whose values range from 0 to 255, where 2 is the darkest possible value and 255 is the brightest possible value.
   - **Pure white is [255,255,255] in [R, G, B] image.**
2. #### **Code.**
   
   - For more details in `scripts/02.[Project] Finding Lane Lines/4_color_selection.py`
   - Eventually, I found that with **`red_threshold = green_threshold = blue_threshold = 200`,** I get a pretty good result, where I can clearly see the lane lines, but most everything else is blacked out.
   - At this point, however, it would still be tricky to extract the exact lines automatically, because we **still have many other pixels detected around the periphery**.

### 2.3 Region masking

#### **Focus on just the region of the image that interests us.**

1. In this case, I'll assume that the front facing camera that took the image is mounted in a fixed position on the car, such that the lane lines will always appear in the same general region of the image. 

   <img src=assets/2_3_1.png width=400>

2. Next, I'll take advantage of this by adding a criterion (a triangular region) to only consider pixels for color selection in the region where we expect to find the lane lines.

#### **Code:**  

- `scripts/02.[Project] Finding Lane Lines/7_color_region.py`

### 2.4 Question: How to find lines of any color?

As it happens, lane lines are not always the same color, and even lines of the same color under different lighting conditions (day, night, etc) may fail to be detected by our simple color selection.

What we need is to take our algorithm to the next level to detect lines of any color **using sophisticated computer vision methods.**



### 2.5 What's Computer Vision

- **Computer Vision:** using algorithms to let a computer to see the world like we see it. Full of depth, and color, and shapes, and meaning.

- Throughout this Nanodegree Program, we will be using **Python** with **OpenCV** for computer vision work. OpenCV stands for Open-Source Computer Vision. For now, you don't need to download or install anything, but later in the program we'll help you get these tools installed on your own computer.

- OpenCV contains extensive libraries of functions that you can use. The OpenCV libraries are well documented, so if you’re ever feeling confused about what the parameters in a particular function are doing, or anything else, you can **find a wealth of information at** [opencv.org](http://opencv.org/).

### 2.6 Canny Edge Detection

The goal of edge detection is to identify the boundaries of an object in an image.****	

#### **Core Operation:**

- convert to grayscale. (in grayscale image, rapid changes in brightness are where the edges.)
- compute the gradient.
- with the brightness of each pixel corresponds to the **strength of the gradient** at that point.
- Finding the edges by tracing out the pixels that follow the strongest gradients.

By identifying edges, we can more easily detect object by their shape.

#### **Image:**

image is a mathematical function of x and y
$$
f(x,y) = \text{pixel value}
$$
*The grascale image is an 8-bit image, so the pixel value range from 0 to 255.*

#### **Take derivative of an image:**

*a measure of change of the image function*
$$
\frac{df}{dx} = \Delta(\text{pixe value})
$$

#### **OpenCV Canny Function:**

The **strength of an edge** is defined by how different the values are in adjacent pixels in the image. <==> **strength of the gradient.**

```python
'''Find the individual pixels that follow the strongest gradients.
input:
 - gray: input grayscale image
 - low_threshold,
 - high_threshold: determine how strong the edges must to be detected.
output:
 - another image
'''
edges = cv2.Canny(gray, low_threshold, high_threshold)
```

- The algorithm will first detect strong edge (strong gradient) pixels above the `high_threshold`, and reject pixels below the `low_threshold`. 
- Next, pixels with values between the `low_threshold` and `high_threshold` will be included as long as they are connected to strong edges. 
- The output `edges` is a binary image with white pixels tracing out the detected edges and black everywhere else. 
- See the [OpenCV Canny Docs](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html) for more details.

Note:

1. The pixel value range implies that derivatives (essentially, the value differences from pixel to pixel) will be on the scale of tens or hundreds. So, **a reasonable range for your threshold parameters would also be in the tens to hundreds**.
2. As far as a ratio of `low_threshold` to `high_threshold`, [John Canny himself recommended](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html#steps) a low to high ratio of 1:2 or 1:3.
3. We'll also include Gaussian smoothing, before running `Canny`, which is essentially a way of **suppressing noise and spurious gradients by averaging** (check out the [OpenCV docs for GaussianBlur](http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur)). 
4. `cv2.Canny()` actually applies Gaussian smoothing internally, but we include it here because you can get a different result by applying further smoothing (and it's not a changeable parameter within `cv2.Canny()`!).
5. You can choose the `kernel_size` for Gaussian smoothing to be any **odd number**. A larger `kernel_size` implies averaging, or smoothing, over a larger area. The example in the previous lesson was `kernel_size = 3`.

#### **Code:**  

- `scripts/02.[Project] Finding Lane Lines/12_canny_edges.py`



### 2.7 Hough Transform

*To find lines, we need to first adopt a model of a line.* 

#### **The Hough transform:** 

- is just a conversion from image space to Hough space.

<img src=assets/2_7_1.png width=400 >

#### **The properties of Hough transform:**

- **A line** in Image Space will be **a single point** at the position (m,b) in Hough Space.
- **Two parallel lines** in image space correspond to **two points in the same column**  in Hough space
- **A point** in image space correspond to **a line** in Hough space. $y_0=mx_0+b$ ==> $b=-x_0 m+y_0$
- **Two points** in image space correspond to **two intersecting lines** in Hough space.
- **The intersection point of the two lines in Hough space** correspond to  **A line in image space that passes through both (x1, y1) and (x2, y2)** in image space?

#### **Results：**

- ***So our strategy to find lines in image space will be look for intersecting lines in Hough space！***

- We do this by dividing up our Hough space into a grid, and define intersecting lines as all lines passing through a given grid cell.

  <img src=assets/2_7_2.png width=400 >

- **Problem:** The vertical lines in image space have infinite slope. 

#### **Solution: a new parameterization **

- Redefine the line in polar coordinates.

  <img src=assets/2_7_3.png width=400 >

- Each point in image space corresponds to a sine curve in Hough space.
  $$
  \rho = \sqrt{x^2+y^2}\sin(\theta+\phi)
  $$
  

  See the [辅助角公式推导](https://zhuanlan.zhihu.com/p/34719623) for more details.

- A whole line points translates into a whole bunch of sine curves in Hough space. The intersection is the parameter of the line.

  <img src=assets/2_7_4.png width=400 >

  

#### Implementing a Hough Transform on Edge Detected Image

To accomplish the task of finding lane lines, we need to specify some parameters to say **what kind of lines we want to detect** (i.e., long lines, short lines, bendy lines, dashed lines, etc.).

- To do this, we'll be using an OpenCV function called `HoughLinesP` that takes several parameters.  (for a look at coding up a Hough Transform from scratch, check [this](https://alyssaq.github.io/2014/understanding-hough-transform/) out.) 

```python
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                             min_line_length, max_line_gap)
```

- In this case, we are operating on the image `masked_edges` (the output from `Canny`) 
- and the output from `HoughLinesP` will be `lines`, which will simply be an array containing the endpoints (x1, y1, x2, y2) of all **line segments detected by the transform operation**. 
- The other parameters define just what kind of line segments we're looking for.
  - First off, `rho` and `theta` are the distance and angular resolution of our grid in Hough space. Remember that, in Hough space, we have a grid laid out along the (Θ, ρ) axis. You need to specify `rho` **in units of pixels** and `theta` **in units of radians.**
  - So, what are reasonable values? Well, rho takes a minimum value of 1, and a reasonable starting place for theta is 1 degree (pi/180 in radians). 
  - The `threshold` parameter specifies the **minimum number of votes (intersections in a given grid cell) a candidate line** needs to have to make it into the output. 
  - The empty `np.array([])` is just a placeholder, no need to change it.
  -  `min_line_length` is the minimum length of a line (in pixels) that you will accept in the output, 
  - and `max_line_gap` is the maximum distance (again, in pixels) between segments that you will allow to be connected into a single line. 

#### Code

See `scripts/02.[Project] Finding Lane Lines/15_hough_transform.py` for more details.

### 2.8 Project Intro

Your goal, in this project, is to write a software pipeline to identify and track the position of the lane lines in a video stream.	

To get started on the project, download or `git clone` [the project repository on GitHub](https://github.com/udacity/CarND-LaneLines-P1). You'll find the instructions in the README file. 



**See `projects/P1_Lane_lines` for my project implementation.**

### 2.9 Starter Kit Installation

In this term, you'll use Python 3 for programming quizzes, labs, and projects. The following will guide you through setting up the programming environment on your local machine.

There are two ways to get up and running:

1. Anaconda
2. Docker

We recommend you first try setting up your environment with Anaconda. It's faster to get up and running and has fewer moving parts.

If the Anaconda installation gives you trouble, try Docker instead.

Follow the instructions in [this README](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md).  **You can also find it in `Term1_Starter_Kit`** folder.

Here is a great link for learning more about [Anaconda and Jupyter Notebooks](https://classroom.udacity.com/courses/ud1111)

### 2.10 Run Some Code

Now that everything is installed, let's make sure it's working!

1. Clone and navigate to the starter kit test repository.

   ```sh
   # NOTE: This is DIFFERENT from  https://github.com/udacity/CarND-Term1-Starter-Kit.git
   git clone https://github.com/udacity/CarND-Term1-Starter-Kit-Test.git
   cd CarND-Term1-Starter-Kit-Test
   ```

2. Launch the Jupyter notebook with Anaconda or Docker. This notebook is simply to make sure the installed packages are working properly.

   The instructions for the first project are on the next page.

   ```sh
   conda activate carnd-term1 # If currently deactivated, i.e. start of a new terminal session
   jupyter notebook test.ipynb
   ```

Docker   
   ```

   ```sh
   # Docker
   docker run -it --rm -p 8888:8888 -v ${pwd}:/src udacity/carnd-term1-starter-kit test.ipynb
   # OR
   docker run -it --rm -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit test.ipynb
   ```

3. Go to [`http://localhost:8888/notebooks/test.ipynb`](http://localhost:8888/notebooks/test.ipynb) in your browser and run all the cells. Everything should execute without error.

#### Troubleshooting

**ffmpeg**

**NOTE:** If you don't have `ffmpeg` installed on your computer you'll have to install it for `moviepy` to work. If this is the case you'll be prompted by an error in the notebook. You can easily install `ffmpeg` by running the following in a code cell in the notebook.

```python
import imageio
imageio.plugins.ffmpeg.download()
```

Once it's installed, `moviepy` should work.

**Docker**

To get the latest version of the [docker image](https://hub.docker.com/r/udacity/carnd-term1-starter-kit/), you may need to run:

```sh
docker pull udacity/carnd-term1-starter-kit
```

Warning! The image is ~2GB!

### 2.11 Project Expectations

For each project in Term 1, keep in mind a few key elements:

- rubric
- code
- writeup
- submission

#### Rubric

Each project comes with a rubric detailing the requirements for passing the project. Project reviewers will check your project against the rubric to make sure that it meets specifications.

Before submitting your project, compare your submission against the rubric to make sure you've covered each rubric point.

Here is an example of a project rubric:

![img](./assets/2_11_1.png)



#### Code

Every project in the term includes code that you will write. For some projects we provide code templates, **often in a Jupyter notebook**. For other projects, there are no code templates.

In either case, you'll need to submit your code files as part of the project. Each project has specific instructions about what files are required. Make sure that your code is commented and easy for the project reviewers to follow.

For the Jupyter notebooks, sometimes you must run all of the code cells and then export the notebook as an HTML file. The notebook will contain instructions for how to do this.

***Because running the code can take anywhere from several minutes to a few hours, the HTML file allows project reviewers to see your notebook's output without having to run the code.***



#### Writeup

All of the projects in Term 1 require a writeup. The writeup is your chance to explain how you approached the project.

It is also an opportunity to show your understanding of key concepts in the program.

We have provided writeup templates for every project so that it is clear what information needs to be in each writeup. These templates can be found in each project repository, with the title `writeup_template.md`.

Your writeup report should explain how you satisfied each requirement in the project rubric.

The writeups can be turned in either as Markdown files (.md) or PDF files.



#### Submission

When submitting a project, you can either submit it as a link to a [GitHub repository](https://github.com/) or as a ZIP file. When submitting a GitHub repository, we advise creating a new repository, specific to the project you are submitting.

GitHub repositories are a convenient way to organize your projects and display them to the world. A GitHub repository also has a README.md file that opens automatically when somebody visits your GitHub repository link.

As a suggestion, the README.md file for each repository can include the following information:

- an overview of the project
- a list of files contained in the repository with a brief description of each file
- any instructions someone might need for running your code

Here is an example of a README file:



![img](./assets/2_11_2.png)

**Related resources:**

- If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).
- To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well.
- GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.



## 3. Introduction to Neural Network

### 3.1 Starting Machine Learning

#### Linear Regression

<img src=./assets/3_1_1.png width=400>

**So how do we fitting this line?**

- Model: y=wx+b
- Error: mse
- Optimization: gradient descent to find minimum error.

**Linear regression** helps predict values **on a continuous spectrum**, like predicting what the price of a house will be.

Linear regression will lead to neural networks, which is a much more advanced classification tool.

**Quiz?**

How about classifying data among discrete classes?

Here are examples of classification tasks:

- Determining whether a patient has cancer
- Identifying the species of a fish
- Figuring out who's talking on a conference call

Classification problems are important for self-driving cars. Self-driving cars might need to classify whether an object crossing the road is a car, pedestrian, and a bicycle. Or they might need to identify which type of traffic sign is coming up, or what a stop light is indicating.



#### Logistic Regression

In this video, Luis demonstrates a classification algorithm called "logistic regression". He uses logistic regression to predict whether a student will be accepted to a university.

<img src=./assets/3_1_2.png width=400>

**So how to find the line that best cuts the data?**

- Model: y=wx+b
- Error (capture the number of errors): log loss function
- Optimization: GD

#### Neural Network

To tackle more realistic data, we use a neural network.

<img src=./assets/3_1_3.png width=400>

<img src=./assets/3_1_4.png width=400>

- input layer -> hidden layer -> output layer
- You can also add more nodes in the middle or even more layers of nodes to map more complex areas in the plane or even in three dimensions or higher dimensional spaces.



#### Perceptron

- Data, like test scores and grades, are fed into a network of interconnected nodes. ***These individual nodes are called [perceptrons](https://en.wikipedia.org/wiki/Perceptron), or artificial neurons***, and they are the basic unit of a neural network. 

- *Each one looks at input data and decides how to categorize that data.* 

**Weights**

- When input comes into a perceptron, it gets multiplied by a weight value that is assigned to this particular input. 
- These weights start out as random values, and as the neural network network learns more about what kind of input data leads to a student being accepted into a university, the network adjusts the weights based on any errors in categorization that results from the previous weights. 
- This is called **training** the neural network.
- A higher weight means the neural network considers that input more important than other inputs, and lower weight means that the data is considered less important. 

**Summing the Input Data**

- Each input to a perceptron has an associated weight that represents its importance. 
- In the next step, the weighted input data are summed to produce a single value, that will help determine the final output 

<img src=./assets/3_1_5.jpeg width=400 >

**Calculating the Output with an Activation Function**

- Finally, the result of the perceptron's summation is turned into an output signal! This is done by feeding the linear combination into an **activation function**.

- One of the simplest activation functions is the **Heaviside step function**. This function returns a **0** if the linear combination is less than 0. It returns a **1** if the linear combination is positive or equal to zero. The [Heaviside step function](https://en.wikipedia.org/wiki/Heaviside_step_function) is shown below, where h is the calculated linear combination:

<img src=assets/3_1_6.png width=300 >

- A bias, represented in equations as *b*, lets us move values in one direction or another. the bias can also be updated and changed by the neural network during training. So after adding a bias, we now have a complete **perceptron formula:**

  <img src=assets/3_1_7.gif width=400 >



Then the neural network starts to learn! Initially, the weights ( *w**i*) and bias (*b*) are assigned a random value, and then they are updated using a learning algorithm like **gradient descent.** The weights and biases change so that the next training example is more accurately categorized, and patterns in data are "**learned**" by the neural network.

#### AND Perceptron Code

- See `scripts/03.Introduction to Neural Network/10_AND_Perceptron_Quiz.py` for more details.
- Tip: Find the parameters of the line to correctly separate all points. 

####  OR Perceptron Quiz

<img src=assets/3_1_8.png width=400 >

- There are two ways of creating an OR perceptron from an AND perception
  - Decrease both two weights ($w_1x+w_2y+b=0$ <==> $y=-w_1/w_2 -b/w_2$)
  - Increase the magnitude of the bias.
  - 两个方案本质上都是减小截距来平移直线。

#### NOT Perceptron Code

The NOT operations only cares about one input. The operation returns a `0` if the input is `1` and a `1` if it's a `0`. The other inputs to the perceptron are ignored.

- See `scripts/03.Introduction to Neural Network/11_NOT_Perceptron_Quiz.py` for more details.

#### XOR Perceptron

An XOR perceptron is a logic gate that outputs `0` if the inputs are the same and `1` if the inputs are different. Unlike previous perceptrons, this graph isn't linearly separable. To handle more complex problems like this, we can chain perceptrons together.

<img src=assets/3_1_9.png width=300 >

**Let's build a multi-layer perceptron from the AND, NOT and OR perceptrons to create XOR logic!**

<img src=assets/3_1_10.png width=300 >

| Perceptron | Operators |
| ---------- | --------- |
| A          | AND       |
| B          | OR        |
| C          | NOT       |

### 3.2 The simplest neural network

So far you've been working with perceptrons where the output is always one or zero. The input to the output unit is passed through an activation function, *f*(*h*), in this case, the step function.

<img src=assets/3_1_6.png width=300 >

The diagram below shows a simple network. The linear combination of the weights, inputs, and bias form the input *h*, which passes through the activation function *f*(*h*), giving the final output of the perceptron, labeled *y*.

<img src=assets/3_2_1.png width=300 >

The cool part about this architecture, and what makes neural networks possible, is that the activation function, *f*(*h*) can be *any function*, not just the step function shown earlier.

- For example, if you let *f*(*h*)=*h*, the output will be the same as the input. Now the output of the network is$y=∑_iw_i*x_i+b$ This equation should be familiar to you, it's the same as the **linear regression model!**

- Other activation functions you'll see are the **logistic** (often called the **sigmoid**), **tanh**, and **softmax** functions. We'll mostly be using the sigmoid function for the rest of this lesson: 
  $$
  sigmoid(x)=1/(1+e^{−x})
  $$
  

  The sigmoid function is bounded between 0 and 1, and as an output can be interpreted as **a probability for success**. It turns out, again, using a sigmoid as the activation function results in the same formulation as **logistic regression**.

***This is where it stops being a perceptron and begins being called a neural network.*** 

#### Simple network Code

For the weights sum, you can use NumPy's [dot product function](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html).

- See `scripts/03.Introduction to Neural Network/13_The_Simplest_Neural_Network.py` for more details.



### 3.3 Gradient Descent

#### Learning weights

We want the network to make predictions as close as possible to the real values. To measure this, we need a metric of how wrong the predictions are, the **error**. A common metric is the **sum of the squared errors** (SSE):
$$
E=\frac{1}{2}∑_μ∑_j[y_j^μ−\hat{y}_j^μ]^2
$$


Our goal is to find weights *w**i**j* that minimize the squared error *E*. To do this with a neural network, typically you'd use **gradient descent**.

*Gradient* is another term for rate of change or slope. If you need to brush up on this concept, check out Khan Academy's [great lectures](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/gradient-and-directional-derivatives/v/gradient) on the topic.

Gradient descent steps to the lowest error:

<img src=assets/3_3_1.png width=300 >

#### Caveats

Since the weights will just go where ever the gradient takes them, **they can end up where the error is low, but not the lowest.** These spots are called local minima. If the weights are initialized with the wrong values, gradient descent could lead the weights into a local minimum, illustrated below.

<img src=assets/3_3_2.png width=300 >

There are methods to avoid this, such as using [momentum](http://sebastianruder.com/optimizing-gradient-descent/index.html#momentum).

#### Gradient descent math

<img src=assets/3_3_3.png width=500 >

*注意：上图的推导少了一个负号，求导时-y的负号漏掉了*

#### Gradient descent Code

- See `scripts/03.Introduction to Neural Network/16_Gradient_Descent_Code.py` for more details.

#### Implementing Gradient Descent Code

Okay, now we know how to update our weights
$$
Δw_{ij}=ηδ_jx_i,
$$
how do we translate this into code?

As an example, I'm going to have you use gradient descent to train a network on **graduate school admissions data** (found at https://stats.idre.ucla.edu/stat/data/binary.csv). This dataset has three input features: **GRE score**, **GPA**, and **the rank of the undergraduate school** (numbered 1 through 4). Institutions with rank 1 have the highest prestige, those with rank 4 have the lowest.

<img src=assets/3_3_4.png width=400>



The goal here is to predict if a student will be admitted to a graduate program based on these features. For this, we'll use a network with one output layer with one unit. We'll use a sigmoid function for the output unit activation.

**Data cleanup**

- The `rank` feature is categorical, the numbers don't encode any sort of relative values. Rank 2 is not twice as much as rank 1, rank 3 is not 1.5 more than rank 2. Instead, we need to use [dummy variables](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)) to encode `rank`, **splitting the data into four new columns** encoded with ones or zeros. Rows with rank 1 have one in the rank 1 dummy column, and zeros in all other columns. Rows with rank 2 have one in the rank 2 dummy column, and zeros in all other columns. And so on. **(one-hot)**

- We'll also need to **standardize** the GRE and GPA data, which means to scale the values such they have zero mean and a standard deviation of 1. *This is necessary because the sigmoid function squashes really small and really large inputs.* The gradient of really small and large inputs is zero, which means that the gradient descent step will go to zero too. 

- This is just a brief run-through, you'll learn more about preparing data later. If you're interested in how I did this, check out the `data_prep.py` file.

<img src=assets/3_3_5.png width=400>

- Now that the data is ready, we see that there are **six input features**: `gre`, `gpa`, and the four `rank` dummy variables.

**Mean Square Error**
$$
E = \frac{1}{2m}\sum_u(y^u-\hat{y}^u)^2
$$
Here's the general algorithm for updating the weights with gradient descent:

- Set the weight step to zero: $Δw_i=0$
- For each record in the training data:
  - Make a forward pass through the network, calculating the output $\hat{y}=f(∑_iw_ix_i)$
  - Calculate the error gradient in the output unit, $δ=(y−\hat{y})∗f′(∑_iw_ix_i)$
  - Update the weight step $Δw_i=Δw_i+δx_i$ (里面已经包含了梯度的负号)
- Update the weights $wi=w_i+ηΔw_i/m$ where *η* is the learning rate and *m* is the number of records. Here we're averaging the weight steps to help reduce any large variations in the training data.
- Repeat for *e* epochs.

1. You can also update the weights on each record instead of averaging the weight steps after going through all the records.
2. Remember that we're using the sigmoid for the activation function, $f(h)=1/(1+e^{−h)}$
3. And the gradient of the sigmoid is $f′(h)=f(h)(1−f(h))$ where *h* is the input to the output unit

**Implementing with Numpy:**

```python
weights = np.random.normal(scale=1/n_features**.5, size=n_features)
```

- scale parameter keeps the input to the sigmoid low for increasing numbers of input units.

**Programming exercise**

- See `scripts/03.Introduction to Neural Network/17_Implementing_Gradient_Descent` for more details.

### 3.4 Multilayer Perceptron

<img src=assets/3_4_1.png width=400>

#### Making a column vector

You see above that sometimes you'll want a column vector, even though by default Numpy arrays work like row vectors. It's possible to get the transpose of an array like so `arr.T`, but for a 1D array, the transpose will return a row vector. Instead, use `arr[:,None]` to create a column vector:

```python
print(features)
> array([ 0.49671415, -0.1382643 ,  0.64768854])

print(features.T)
> array([ 0.49671415, -0.1382643 ,  0.64768854])

print(features[:, None])
> array([[ 0.49671415],
       [-0.1382643 ],
       [ 0.64768854]])
```

Alternatively, you can create arrays with two dimensions. Then, you can use `arr.T` to get the column vector.

```python
np.array(features, ndmin=2)
> array([[ 0.49671415, -0.1382643 ,  0.64768854]])

np.array(features, ndmin=2).T
> array([[ 0.49671415],
       [-0.1382643 ],
       [ 0.64768854]])
```

I personally prefer keeping all vectors as 1D arrays, it just works better in my head.

#### Code

Below, you'll implement a forward pass through a 4x3x2 network, with sigmoid activation functions for both layers.

Things to do:

- Calculate the input to the hidden layer.
- Calculate the hidden layer output.
- Calculate the input to the output layer.
- Calculate the output of the network.



- See `scripts/03.Introduction to Neural Network/18_Multilayer_Perception` for more details.

### 3.5 Backpropagation

<img src=assets/3_5_1.png width=400>

- 每个节点传播过来的误差等于上一层和它有连接的所有节点的误差的加权和（权重为二者的连接权重）然后乘以本节点的激活函数对激活值的梯度

- 根据此法则，将误差从输出层逐层向前传播

- 最后每层更新的参数值为学习率乘以本层误差乘以本层输出值（激活值或输入值）：
  $$
  \Delta w_{ij} = \eta \delta_{j}x_i
  $$
  

#### **Working through an example**

<img src=assets/3_5_2.png height=200>

<img src=assets/3_5_3.png >

- From this example, you can see one of the effects of using the sigmoid function for the activations. The maximum derivative of the sigmoid function is 0.25, so the errors in the output layer get reduced by at least 75%, and errors in the hidden layer are scaled down by at least 93.75%! You can see that if you have a lot of layers, using a sigmoid activation function will quickly reduce the weight steps to tiny values in layers near the input. This is known as the **vanishing gradient** problem.

#### Code

If you multiply a row vector array with a column vector array, it will multiply the first element in the column by each element in the row vector and set that as the first row in a new 2D array. This continues for each element in the column vector, so you get a 2D array that has shape `(len(column_vector), len(row_vector))`.

```python
hidden_error*inputs[:,None]
```

- **See `scripts/03.Introduction to Neural Network/19_Backpropagation` for more details.**
- **See `scripts/03.Introduction to Neural Network/20_Implementing_Backpropagation` for more details.**

### 3.6 Further Reading

Backpropagation is fundamental to deep learning. TensorFlow and other libraries will perform the backprop for you, but you should really *really* understand the algorithm. We'll be going over backprop again, but here are some extra resources for you:

- From Andrej Karpathy: [Yes, you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.vt3ax2kg9)
- Also from Andrej Karpathy, [a lecture from Stanford's CS231n course](https://www.youtube.com/watch?v=59Hbtz7XgjM)



## 4. MiniFlow

### 4.1 Introduction

In this lab, you’ll build a library called **`MiniFlow`** which will be your own version of [TensorFlow](http://tensorflow.org/)! [(link for China)](http://tensorfly.cn/)

- TensorFlow is one of the most popular open source neural network libraries, built by the team at Google Brain over just the last few years.
- Following this lab, you'll spend the remainder of this module actually working with open-source deep learning libraries like [TensorFlow](http://tensorflow.org/) and [Keras](https://keras.io/). So why bother building MiniFlow? 

- The goal of this lab is to demystify two concepts at the heart of neural networks - **backpropagation** and **differentiable graphs**.
  - Backpropagation is the process by which neural networks update the weights of the network over time. (You may have seen it in [this video](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/83a4e710-a69e-4ce9-9af9-939307c0711b/concepts/45cebfff-236d-453a-be24-7a179c1fa8ed) earlier.)
  - Differentiable graphs are graphs where the nodes are [differentiable functions](https://en.wikipedia.org/wiki/Differentiable_function). They are also useful as *visual aids* for understanding and calculating complicated derivatives. This is the fundamental abstraction of TensorFlow - it's a framework for creating differentiable graphs.

With graphs and backpropagation, you will be able to create your own nodes and properly compute the derivatives. Even more importantly, you will be able to think and reason in terms of these graphs.

### 4.2 Graphs

**What is a Neural Network?**

- A neural network is a graph of mathematical functions such as [linear combinations](https://en.wikipedia.org/wiki/Linear_combination) and activation functions. The graph consists of **nodes**, and **edges**.

- Nodes in each layer (except for nodes in the input layer) perform mathematical functions using inputs from nodes in the previous layers. For example, a node could represent *f*(*x*,*y*)=*x*+*y*, where *x* and *y* are input values from nodes in the previous layer.
- Layers between the input layer and the output layer are called **hidden layers**.
- The edges in the graph describe the connections between the nodes, along which the values flow from one layer to the next. *These edges can also apply operations to the values that flow along them, such as multiplying by weights, adding biases, etc..* 
- MiniFlow won't use a special class for edges. Instead, its nodes will perform both their own calculations and those of their input edges. This will be more clear as you go through these lessons.

**Forward Propagation:**

By propagating values from the first layer (the input layer) through all the mathematical functions represented by each node, the network outputs a value. This process is called a **forward pass**.

**Graphs**

There are generally two steps to create neural networks:

1. Define the graph of nodes and edges.
2. Propagate values through the graph.

`MiniFlow` works the same way. You'll define the nodes and edges of your network with one method and then propagate values through the graph with another method.

### 4.3 MiniFlow Architecture

Let's consider how to implement this **graph structure** in `MiniFlow`. We'll use a Python class to represent a generic node.

```python
class Node(object):
    def __init__(self):
        # Properties will go here!
```

We know that each node might receive input from multiple other nodes. We also know that each node **creates a single output**, which will likely be passed to other nodes. Let's add two lists: 

- one to store references to the inbound nodes, 
- and the other to store references to the outbound nodes.

```python
class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        self.outbound_nodes = []
        # For each inbound Node here, add this Node as an outbound Node to _that_ Node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
```

Each node will eventually calculate a value that represents its output. Let's initialize the `value` to `None` to indicate that it exists but hasn't been set yet.

```python
class Node(object):
    def __init__(self, inbound_nodes=[]):
        ...

        self.value = None
```

Each node will need to be able to pass values forward and perform backpropagation (more on that later). For now, let's add a placeholder method for forward propagation. We'll deal with backpropagation later on.

```python
class Node(object):
    ...

    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplemented
```

#### Nodes that Calculate

While `Node` defines the base set of properties that every node holds, only specialized [subclasses](https://docs.python.org/3/tutorial/classes.html#inheritance) of `Node` will end up in the graph. As part of this lab, you'll build the subclasses of `Node` that can perform calculations and hold values. For example, consider the `Input` subclass of `Node`.

```python
class Input(Node):
    def __init__(self):
        # An Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator.
        Node.__init__(self)

    # NOTE: Input node is the only node where the value
    # may be passed as an argument to forward().
    #
    # All other node implementations should get the value
    # of the previous node from self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value
```

Unlike the other subclasses of `Node`, the `Input` subclass does not actually calculate anything. The `Input` subclass just holds a `value`, such as a data feature or a model parameter (weight/bias).

You can set `value` either explicitly or with the `forward()` method. This value is then fed through the rest of the neural network.

#### The Add Subclass

`Add`, which is another subclass of `Node`, actually can perform a calculation (addition).

```python
class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        """
        You'll be writing code here in the next quiz!
        """
```

Notice the difference in the `__init__` method, `Add.__init__(self, [x, y])`. Unlike the `Input` class, which has no inbound nodes, the `Add` class takes 2 inbound nodes, `x` and `y`, and adds the values of those nodes.

### 4.4 Forward Propagation

#### Topological sort

`MiniFlow` has two methods to help you define and then run values through your graphs: `topological_sort()` and `forward_pass()`.

<img src=assets/4_4_1.jpeg width=400>

In order to define your network, you'll need to define **the order of operations** for your nodes. Given that the input to some node depends on the outputs of others, you need to flatten the graph in such a way where all the input dependencies for each node are resolved before trying to run its calculation. This is a technique called a [topological sort](https://en.wikipedia.org/wiki/Topological_sorting).

- The `topological_sort()` function implements topological sorting using [Kahn's Algorithm](https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm). The details of this method are not important, the result is; `topological_sort()` returns **a sorted list of nodes in which all of the calculations can run in series**.
- `topological_sort()` takes in a `feed_dict`, which is how we initially set a value for an `Input` node. The `feed_dict` is represented by the Python dictionary data structure. Here's an example use case:

```python
# Define 2 `Input` nodes.
x, y = Input(), Input()

# Define an `Add` node, the two above`Input` nodes being the input.
add = Add(x, y)

# The value of `x` and `y` will be set to 10 and 20 respectively.
feed_dict = {x: 10, y: 20}

# Sort the nodes with topological sort.
sorted_nodes = topological_sort(feed_dict=feed_dict)
```

- You can find the source code for `topological_sort()` in miniflow.py in the programming quiz below.

#### Forward pass

The other method at your disposal is `forward_pass()`, which actually runs the network and outputs a value.

```python
def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: The output node of the graph (no outgoing edges).
        `sorted_nodes`: a topologically sorted list of nodes.

    Returns the output node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value
```

#### Code - Passing Values Forward

Create and run this graph!

<img src=assets/4_4_2.png width=200 >

**Setup:**

Review `nn.py` and `miniflow.py`.

The neural network architecture is already there for you in nn.py. It's your job to finish `MiniFlow` to make it work.

For this quiz, I want you to:

1. Open `nn.py` below. **You don't need to change anything.** I just want you to see how `MiniFlow` works.
2. Open `miniflow.py`. **Finish the `forward` method on the `Add` class. All that's required to pass this quiz is a correct implementation of `forward`.**
3. Test your network by hitting "Test Run!" When the output looks right, hit "Submit!"



- **See `scripts/04.MiniFlow/05_Forward_Propagation` for more details.**

- **See `scripts/04.MiniFlow/06_Forward_Propagation` for more details.**



### 4.5 Learning and Loss

The output, *o*, is just the weighted sum of the inputs plus the bias:
$$
o = \sum_i x_iw_i + b
$$
Remember, by varying the weights, you can vary the amount of influence any given input has on the output. The learning aspect of neural networks takes place during a process known as backpropagation. In backpropogation, the network modifies the weights to improve the network's output accuracy. You'll be applying all of this shortly.

In this next quiz, you'll try to build a linear neuron that generates an output by applying a simplified version of Equation (1). `Linear` should take an list of inbound nodes of length *n*, a list of weights of length *n*, and a bias.

**Instructions**

1. Open nn.py below. Read through the neural network to see the expected output of `Linear`.
2. Open miniflow.py below. Modify `Linear`, which is a subclass of `Node`, to generate an output with Equation (1).

#### Code

- **See `scripts/04.MiniFlow/07_Linear` for more details.**



### 4.6 Linear Transform

$$
Z = X W + b
$$

- X is now an `m by n` matrix. Each row has n inputs/features.
- W is now a `n by k` matrix.	

- b is a row matrix of biases, one for each output. `1 by k`

*Equation (2) can also be viewed as Z = XW + B where B is the biases vector, b, stacked m times as a row. Due to broadcasting it's abbreviated to Z = XW + b.*



I want you to rebuild `Linear` to handle matrices and vectors using the venerable Python math package `numpy` to make your life easier.

I used `np.array` ([documentation](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html)) to create the matrices and vectors. You'll want to use `np.dot`, which functions as matrix multiplication for 2D arrays ([documentation](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html)), to multiply the input and weights matrices from Equation (2). It's also worth noting that numpy actually overloads the `__add__` operator so you can use it directly with `np.array` (eg. `np.array() + np.array()`).

**Instructions:**

1. Open nn.py. See how the neural network implements the `Linear` node.
2. Open miniflow.py. Implement Equation (2) within the forward pass for the `Linear` node.
3. Test your work!

#### Code

- **See `scripts/04.MiniFlow/08_Linear_Transform` for more details.**



### 4.7 Sigmoid Function

Linear transforms are great for simply *shifting* values, but neural networks often require a more nuanced transform. 

- For instance, one of the original designs for an artificial neuron, [the perceptron](https://en.wikipedia.org/wiki/Perceptron), exhibit binary output behavior. **Perceptrons** compare a weighted input to a threshold. When the weighted input exceeds the threshold, the perceptron is **activated** and outputs `1`, otherwise it outputs `0`.

  You could model a perceptron's behavior as a step function.

  <img src=assets/4_7_1.png width=200 >

- but step functions are not continuous and not differentiable, which is *very bad*. Differentiation is what makes gradient descent possible.

The sigmoid function:
$$
sigmoid(x)= \frac{1}{1+e^{-x}}
$$


replaces thresholding with a beautiful S-shaped curve that mimics the activation behavior of a perceptron while being differentiable. 

<img src=assets/4_7_2.png width=200 >

As a bonus, the sigmoid function has a very simple derivative that that can be calculated from the sigmoid function itself.

$$
\sigma'(x) = \sigma(x)*(1-\sigma(x))
$$

- Conceptually, the sigmoid function makes decisions. When given weighted features from some data, it indicates whether or not the features contribute to a classification. 
- In that way, a sigmoid activation works well following a linear transformation. As it stands right now with random weights and bias, the sigmoid node's output is also random. The process of learning through backpropagation and gradient descent, which you will implement soon, modifies the weights and bias such that activation of the sigmoid node begins to match expected outputs.

#### Code

- **See `scripts/04.MiniFlow/09_Sigmoid` for more details.**



### 4.8 Cost

As you may recall, neural networks improve the **accuracy** of their outputs by modifying weights and biases in response to training against labeled datasets.

People use different names for this accuracy measurement, often terming it **loss** or **cost**. I'll use the term *cost* most often.

For this lab, you will calculate the cost using the mean squared error (MSE). It looks like so:
$$
C(w,b)=\frac{1}{m}\sum_x||y(x)−a||^2
$$

- Here *w* denotes the collection of all weights in the network, *b* all the biases, *m* is the total number of training examples, *a* is the approximation of *y(x)* by the network, both *a* and *y(x)* are vectors of the same length.

The collection of weights is all the weight matrices **flattened into vectors** and concatenated to one big vector. The same goes for the collection of biases except they're already vectors so there's no need to flatten them prior to the concatenation.

Here's an example of creating *w* in code:

```python
# 2 by 2 matrices
w1  = np.array([[1, 2], [3, 4]])
w2  = np.array([[5, 6], [7, 8]])

# flatten
w1_flat = np.reshape(w1, -1)
w2_flat = np.reshape(w2, -1)

w = np.concatenate((w1_flat, w2_flat))
# array([1, 2, 3, 4, 5, 6, 7, 8])
```

It's a nice way to abstract all the weights and biases used in the neural network and makes some things easier to write as we'll see soon in the upcoming gradient descent sections.

**NOTE:** It's not required you do this in your code! It's just easier to do this talk about the weights and biases as a collective than consider them invidually.

#### Code

- **See `scripts/04.MiniFlow/10_cost` for more details.**



### 4.9 Gradient Descent

- Gradient descent works by first calculating the slope of the plane at the current point, which includes calculating the partial derivatives of the loss with respect to **all of the parameters**. This set of partial derivatives is called the **gradient**. 

- Technically, the gradient actually points uphill, in the direction of **steepest ascent**. But if we put a `-` sign at the front this value, we get the direction of **steepest descent**, which is what we want.

-  *learning rate* empirically values in the range 0.1 to 0.0001 work well. The range 0.001 to 0.0001 is popular, as 0.1 and 0.01 are sometimes too large.

- Here's the formula for gradient descent (pseudocode):

  ```python
  x = x - learning_rate * gradient_of_x
  ```

#### Code

- **See `scripts/04.MiniFlow/12_Gradient_Descent` for more details.**



### 4.10 Backpropagation

#### Derivatives

In calculus, the derivative tells us how something changes with respect to something else. Or, put differently, how *sensitive* something is to something else.

Let's take the function *f*(*x*)=*x*2 as an example. In this case, the derivative of *f*(*x*) is 2*x*. Another way to state this is, "the derivative of *f*(*x*) with respect to *x* is 2*x*".

Using the derivative, we can say *how much* a change in *x* effects *f*(*x*). For example, when *x* is 4, the derivative is 8 (2*x*=2∗4=8). This means that if *x* is increased or decreased by 1 unit, then *f*(*x*) will increase or decrease by 8. *(the slope (or derivative) itself changes as x changes)*

#### Chain Rule

We simply calculate the derivative of the cost with respect to each parameter in the network. The gradient is a vector of all these derivatives.

In reality, neural networks are a composition of functions, so computing the derivative of the cost w.r.t a parameter isn't quite as straightforward as calculating the derivative of a polynomial function like *f*(*x*)=*x*2. This is where the chain rule comes into play.

*I highly recommend checking out [Khan Academy's lessons on partial derivatives](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivatives/v/partial-derivatives-introduction) and [gradients](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/gradient-and-directional-derivatives/v/gradient) if you need more of a refresher.*

Say we have a new function *f*∘*g*(*x*)=*f*(*g*(*x*)). We can calculate the derivative of *f*∘*g* w.r.t *x* , denoted by applying the chain rule.
$$
\frac{∂f∘g}{∂x}=\frac{∂g}{∂x}\frac{∂f}{∂g}
$$
The way to think about this is:

> In order to know the effect *x* has on *f*, we first need to know the effect *x* has on *g*, and then the effect *g* has on *f*.



Let's now look at a more complex example. Consider the following neural network in MiniFlow:

```python
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
cost = MSE(l2, y)
```

This also can be written as a composition of functions `MSE(Linear(Sigmoid(Linear(X, W1, b1)), W2, b2), y)`. Our goal is to adjust the weights and biases represented by the `Input` nodes `W1, b1, W2, b2`, such that the cost is minimized.

#### Additional Resources

- [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.fowl6fvfk) by Andrej Karpathy
- [Vector, Matrix, and Tensor Derivatives](http://cs231n.stanford.edu/vecDerivs.pdf) by Erik Learned-Miller.

#### Code

- **See `scripts/04.MiniFlow/13_Backpropagation` for more details.**



### 4.11 Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) is a version of Gradient Descent where on each forward pass **a batch of data** is randomly sampled from total dataset. 

A naïve implementation of SGD involves:

1. Randomly sample a batch of data from the total dataset.
2. Running the network forward and backward to calculate the gradient (with data from (1)).
3. Apply the gradient descent update.
4. Repeat steps 1-3 until convergence or the loop is stopped by another mechanism (i.e. the number of epochs).

As a reminder, here's the gradient descent update equation, where *α* represents the learning rate:
$$
x = x - \alpha * \frac{\partial cost}{\partial x}
$$
We're also going to use an actual dataset for this quiz, the [Boston Housing dataset](https://archive.ics.uci.edu/ml/datasets/Housing). After training the network will be able to predict prices of Boston housing!

#### Code

- **See `scripts/04.MiniFlow/14_SGD` for more details.**

