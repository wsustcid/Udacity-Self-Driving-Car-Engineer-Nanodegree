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



## 5. Introduction to Tensorflow

At the end of the this section, you will use the TensorFlow deep library to build your own convolutional neural network.

### 5.1 Introduction  to Deep Neural Networks

<img src=assets/5_1_1.png width=500 >

<img src=assets/5_1_2.png width=500 >

### 5.2 Installing Tensorflow

Throughout this lesson, you'll apply your knowledge of neural networks on real datasets using [TensorFlow](https://www.tensorflow.org/) [(link for China)](http://www.tensorfly.cn/), an open source Deep Learning library created by Google.

You’ll use TensorFlow to classify images from the notMNIST dataset - a dataset of images of English letters from A to J. You can see a few example images below.

Your goal is to automatically detect the letter based on the image in the dataset. You’ll be working on your own computer for this lab, so, first things first, install TensorFlow!

#### OS X, Linux, Windows

**Prerequisites**

*Intro to TensorFlow* requires [Python 3.4 or higher](https://www.python.org/downloads/) and [Anaconda](https://www.continuum.io/downloads). If you don't meet all of these requirements, please install the appropriate package(s).

**Install TensorFlow**

You're going to use an Anaconda environment for this class. If you're unfamiliar with Anaconda environments, check out the [official documentation](http://conda.pydata.org/docs/using/envs.html). More information, tips, and troubleshooting for installing tensorflow on Windows can be found [here](https://www.tensorflow.org/install/install_windows).

Run the following commands to setup your environment:

```sh
conda create --name=IntroToTensorFlow python=3 anaconda
source activate IntroToTensorFlow
conda install -c conda-forge tensorflow
```

That's it! You have a working environment with TensorFlow. Test it out with the code in the *Hello, world!* section below.

#### Docker on Windows

Docker instructions were offered prior to the availability of a stable Windows installation via pip or Anaconda. Please try Anaconda first, Docker instructions have been retained as an alternative to an installation via Anaconda.

**Install Docker**

Download and install Docker from the [official Docker website](https://docs.docker.com/engine/installation/windows/).

**Run the Docker Container**

Run the command below to start a jupyter notebook server with TensorFlow:

```sh
docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow
```

*Users in China should use the `b.gcr.io/tensorflow/tensorflow` instead of `gcr.io/tensorflow/tensorflow`*

You can access the jupyter notebook at [localhost:8888](http://localhost:8888/). The server includes 3 examples of TensorFlow notebooks, but you can create a new notebook to test all your code.

#### Hello, world!

Try running the following code in your Python console to make sure you have TensorFlow properly installed. The console will print "Hello, world!" if TensorFlow is installed. Don’t worry about understanding what it does. You’ll learn about it in the next section.

```python
import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
```

**Errors**

If you're getting the error `tensorflow.python.framework.errors.InvalidArgumentError: Placeholder:0 is both fed and fetched`, you're running an older version of TensorFlow. Uninstall TensorFlow, and reinstall it using the instructions above. For more solutions, check out the [Common Problems](https://www.tensorflow.org/get_started/os_setup#common_problems) section.

### 5.3 TF Basis

#### Input

In TensorFlow, data isn’t stored as integers, floats, or strings. These values are encapsulated in an object called a tensor. In the case of `hello_constant = tf.constant('Hello World!')`, `hello_constant` is a 0-dimensional string tensor, but tensors come in a variety of sizes as shown below:

```python
# A is a 0-dimensional int32 tensor
A = tf.constant(1234) 
# B is a 1-dimensional int32 tensor
B = tf.constant([123,456,789]) 
# C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])
```

[`tf.constant()`](https://www.tensorflow.org/api_docs/python/tf/constant) is one of many TensorFlow operations you will use in this lesson. The tensor returned by [`tf.constant()`](https://www.tensorflow.org/api_docs/python/tf/constant) is called a constant tensor, **because the value of the tensor never changes.**

**Session**

TensorFlow’s api is built around the idea of a **computational graph**, a way of visualizing a mathematical process which you learned about in the MiniFlow lesson. Let’s take the TensorFlow code you ran and turn that into a graph:

<img src=assets/5_2_1.png width=400 >

A "TensorFlow Session", as shown above, is an environment for running a graph. **The session is in charge of allocating the operations to GPU(s) and/or CPU(s),** including remote machines. 

**Input**

In the last section, you passed a tensor into a session and it returned the result. What if you want to use a non-constant? This is where [`tf.placeholder()`](https://www.tensorflow.org/api_docs/python/tf/placeholder) and `feed_dict` come into place. In this section, you'll go over the basics of **feeding data into TensorFlow.**

**tf.placeholder()**

Sadly you can’t just set `x` to your dataset and put it in TensorFlow, because over time you'll want your TensorFlow model to take in different datasets with different parameters. You need [`tf.placeholder()`](https://www.tensorflow.org/api_docs/python/tf/placeholder)!

- [`tf.placeholder()`](https://www.tensorflow.org/api_docs/python/tf/placeholder) returns a tensor that gets its value from data passed to the [`tf.session.run()`](https://www.tensorflow.org/api_docs/python/tf/Session#run) function, allowing you to set the input right before the session runs.

**Session’s feed_dict**

Use the `feed_dict` parameter in [`tf.session.run()`](https://www.tensorflow.org/api_docs/python/tf/Session#run) to set the placeholder tensor. 

```python
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
```

**Note:** If the data passed to the `feed_dict` doesn’t match the tensor type and can’t be cast into the tensor type, you’ll get the error “`ValueError: invalid literal for`...”.



#### Math

Getting the input is great, but now you need to use it. You're going to use basic math functions that everyone knows and loves - add, subtract, multiply, and divide - with tensors. (There's many more math functions you can check out in the [documentation](https://www.tensorflow.org/api_docs/python/math_ops/).)

**Addition**

```python
x = tf.add(5, 2)  # 7
```

You’ll start with the add function. The [`tf.add()`](https://www.tensorflow.org/api_guides/python/math_ops) function does exactly what you expect it to do. It takes in two numbers, two tensors, or one of each, and returns their sum as a tensor.

**Subtraction and Multiplication**

Here’s an example with subtraction and multiplication.

```python
x = tf.subtract(10, 4) # 6
y = tf.multiply(2, 5)  # 10
```

**Converting types**

It may be necessary to convert between types to make certain operators work together. For example, if you tried the following, it would fail with an exception:

```python
tf.subtract(tf.constant(2.0),tf.constant(1))  # Fails with ValueError: Tensor conversion requested dtype float32 for Tensor with dtype int32:
```

That's because the constant `1` is an integer but the constant `2.0` is a floating point value and `subtract` expects them to match.

In cases like these, you can either make sure your data is all of the same type, or you can cast a value to another type. In this case, converting the `2.0` to an integer before subtracting, like so, will give the correct result:

```python
tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))   # 1
```



#### Varibale

In order to use weights and bias, you'll need a Tensor that can be modified. This leaves out [`tf.placeholder()`](https://www.tensorflow.org/api_docs/python/tf/placeholder) and [`tf.constant()`](https://www.tensorflow.org/api_docs/python/tf/constant), since those Tensors can't be modified. This is where [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) class comes in.

**tf.Variable()**

```python
x = tf.Variable(5)
```

The [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) class creates a tensor **with an initial value that can be modified**, much like a normal Python variable. 

- This tensor stores its state in the session, so you must initialize the state of the tensor manually. You'll use the [`tf.global_variables_initializer()`](https://www.tensorflow.org/programmers_guide/variables) function to **initialize the state of all the Variable tensors.**

**Initialization**

```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

- The [`tf.global_variables_initializer()`](https://www.tensorflow.org/programmers_guide/variables) call returns an operation that will initialize all TensorFlow variables from the graph. You call the operation using a session to initialize all the variables as shown above. 
- Initializing the weights with random numbers from a normal distribution is good practice. Randomizing the weights helps the model from becoming stuck in the same place every time you train it. 
- Similarly, choosing weights from a normal distribution prevents any one weight from overwhelming other weights. You'll use the [`tf.truncated_normal()`](https://www.tensorflow.org/api_docs/python/tf/truncated_normal) function to generate random numbers from a normal distribution.

**tf.truncated_normal()**

```python
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
```

- The [`tf.truncated_normal()`](https://www.tensorflow.org/api_docs/python/tf/truncated_normal) function returns a tensor with random values from a normal distribution whose magnitude is no more than 2 standard deviations from the mean.

- Since the weights are already helping prevent the model from getting stuck, you don't need to randomize the bias. Let's use the simplest solution, **setting the bias to 0.**

**tf.zeros()**

```python
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))
```

- The [`tf.zeros()`](https://www.tensorflow.org/api_docs/python/tf/zeros) function returns a tensor with all zeros**.**

**tf.matmul()**

- Since `xW` in `xW + b` is matrix multiplication, you have to use the [`tf.matmul()`](https://www.tensorflow.org/api_docs/python/tf/matmul) function instead of [`tf.multiply()`](https://www.tensorflow.org/api_docs/python/tf/multiply).

#### Softmax

$$
S(y_i) = \frac{e^{y_i}}{\sum_j e^{y_j}}
$$

In the one dimensional case, the array is just a single set of logits. In the two dimensional case, each column in the array is a set of logits. The `softmax(x)` function should return a NumPy array of the same shape as `x`.

- [`tf.nn.softmax()`](https://www.tensorflow.org/api_docs/python/tf/nn/softmax) implements the softmax function for you. It takes in logits and returns softmax activations.

#### **One-hot Encoding**

#### Cross-Entropy

<img src=assets/5_3_1.png width=400 >

**Reduce Sum**

```python
x = tf.reduce_sum([1, 2, 3, 4, 5])  # 15
```

The [`tf.reduce_sum()`](https://www.tensorflow.org/api_docs/python/tf/reduce_sum) function takes an array of numbers and sums them together.

**Natural Log**

```python
x = tf.log(100)  # 4.60517
```

This function does exactly what you would expect it to do. [`tf.log()`](https://www.tensorflow.org/api_docs/python/tf/log) takes the natural log of a number.

#### Logistic Classifier

<img src=assets/5_3_2.png width=400 >

#### Numerical Stability

<img src=assets/5_3_3.png width=400 >

0.953674316406

**Solution:**

- One good guiding principle is that we always want our variables to have 0 mean and equal variance whenever possible.
  $$
  \mu(X_i) = 0 \\
  \sigma(X_i) = \sigma(x_j)
  $$
  <img src=assets/5_3_4.png width=400 >

- For image:
  $$
  pixel = \frac{R/G/B - 128}/128
  $$
  it makes it much easier for the optimization to processed numerically.

#### Normalized Inputs and Initial Weights

<img src=assets/5_3_5.png width=400 >

#### Measuring Performance

- Training Data
- Validation Data
- Test Data

Note:

- 在调试完成之前永远不要接触测试集，如果没有验证集，你根据测试集去调试，你的观察，调试的导向都是在将测试集的信息添加到模型中，这样模型还是没有泛化能力。
- 精度指标的提升也取决于测试集的大小，测试集很大，及时0.1%的提升也是有效的。

#### SGD

<img src=assets/5_3_6.png width=400 >

**Momentum**

- take advantages of the knowledge that we’ve accumulated from previous steps about where we should be heading. 

- A cheap way to do that is to keep a running average of the gradients and to use that running average instead of the direction of the current.
  $$
  M <- 0.9M + \Delta
  $$
  

  

**Learning rate decay**

- apply an exponential decay to the lr
- or make lr smaller every time the loss reaches a plateau.

Note:

较大的学习率，刚开始训练快，但不一代表能训练的好，不要被刚开始的结果欺骗！！

<img src=assets/5_3_7.png width=400 >

**Parameter Hyperspace**

<img src=assets/5_3_8.png width=400 >

#### Mini-batching

Mini-batching is a technique for training on subsets of the dataset instead of all the data at one time. This provides the ability to train a model, even if a computer lacks the memory to store the entire dataset.

It's also quite useful combined with SGD. The idea is to **randomly shuffle the data at the start of each epoch**, then create the mini-batches. For each mini-batch, you train the network weights with gradient descent. Since these batches are random, you're performing SGD with each batch.

Unfortunately, it's sometimes impossible to divide the data into batches of exactly equal size. For example, imagine you'd like to create batches of 128 samples each from a dataset of 1000 samples. Since 128 does not evenly divide into 1000, you'd wind up with 7 batches of 128 samples, and 1 batch of 104 samples. (7*128 + 1*104 = 1000)

In that case, the size of the batches would vary, so you need to take advantage of TensorFlow's [`tf.placeholder()`](https://www.tensorflow.org/api_docs/python/tf/placeholder) function to receive the varying batch sizes.

Continuing the example, if each sample had `n_input = 784` features and `n_classes = 10` possible labels, the dimensions for `features` would be `[None, n_input]` and `labels` would be `[None, n_classes]`.

```python
# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
```

What does `None` do here?

The `None` dimension is a placeholder for the batch size. At runtime, TensorFlow will accept any batch size greater than 0.



### 5.4 AWS GPU Instances

See **Videos/05.Introduction to Tensorflow/35.AWS GPU Instances.html** for more details.

### 5.5 TensorFlow Neural Network Lab

We've prepared a Jupyter notebook that will guide you through the process of creating a single layer neural network in TensorFlow.

**Setup**

If you haven't already setup the Term1 Starter Kit go [here](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/83ec35ee-1e02-48a5-bdb7-d244bd47c2dc/lessons/8c82408b-a217-4d09-b81d-1bda4c6380ef/concepts/4f1870e0-3849-43e4-b670-12e6f2d4b7a7).

**Clone the Repository and Run the Notebook**

Run the commands below to clone the Lab Repository and then run the notebook:

```sh
git clone https://github.com/udacity/CarND-TensorFlow-Lab.git
# Make sure the starter kit environment is activated!
jupyter notebook

# See Projects/TensorFlow_Lab for my implementation
```

**View The Notebook**

Open a browser window and go [here](http://localhost:8888/notebooks/CarND-TensorFlow-Lab/lab.ipynb). This is the notebook you'll be working on. The notebook has 3 problems for you to solve:

- Problem 1: Normalize the features
- Problem 2: Use TensorFlow operations to create features, labels, weight, and biases tensors
- Problem 3: Tune the learning rate, number of steps, and batch size for the best accuracy

## 6. Deep Neural Networks

### 6.1 Introduction to DNN

**Linear Models:**

<img src=assets/6_1_1.png width=400 >

**Network of ReLUs:**

<img src=assets/6_1_2.png width=400 >

- make the function into nonlinear by introducing ReLU, 
- and the number of ReLU  units can be tuned compared to the Logistic Classifier.

**TensorFlow ReLUs:**

A Rectified linear unit (ReLU) is type of [activation function](https://en.wikipedia.org/wiki/Activation_function) that is defined as `f(x) = max(0, x)`. The function returns 0 if `x` is negative, otherwise it returns `x`. TensorFlow provides the ReLU function as [`tf.nn.relu()`](https://www.tensorflow.org/api_docs/python/tf/nn/relu), as shown below.

```python
# Hidden Layer with ReLU activation function
hidden_layer = tf.add(tf.matmul(features, hidden_weights), hidden_biases)
hidden_layer = tf.nn.relu(hidden_layer)

output = tf.add(tf.matmul(hidden_layer, output_weights), output_biases)
```

### 6.2 Deep Neural Network in TensorFlow

You've seen how to build a logistic classifier using TensorFlow. Now you're going to see how to use the logistic classifier to build a deep neural network.

In the following walkthrough, we'll step through TensorFlow code written to classify the letters in the MNIST database. If you would like to run the network on your computer, the file is provided [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58a61a3a_multilayer-perceptron/multilayer-perceptron.zip). You can find this and many more examples of TensorFlow at [Aymeric Damien's GitHub repository](https://github.com/aymericdamien/TensorFlow-Examples).

#### Code

**TensorFlow MNIST**

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)
```

**Learning Parameters**

```python
import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 128  # Decrease batch size if you don't have enough memory
display_step = 1

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
```

The focus here is on the architecture of multilayer neural networks, not parameter tuning, so here we'll just give you the learning parameters.

**Hidden Layer Parameters**

```python
n_hidden_layer = 256 # layer number of features
```

The variable `n_hidden_layer` determines the size of the hidden layer in the neural network. This is also known as the width of a layer.

**Weights and Biases**

```python
# Store layers weight & bias
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
```

Deep neural networks use multiple layers with each layer requiring it's own weight and bias. The `'hidden_layer'` weight and bias is for the hidden layer. The `'out'` weight and bias is for the output layer. If the neural network were deeper, there would be weights and biases for each additional layer.

**Input**

```python
# tf Graph input
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

x_flat = tf.reshape(x, [-1, n_input])
```

The MNIST data is made up of 28px by 28px images with a single [channel](https://en.wikipedia.org/wiki/Channel_(digital_image)). The [`tf.reshape()`](https://www.tensorflow.org/versions/master/api_docs/python/tf/reshape) function above reshapes the 28px by 28px matrices in `x` into row vectors of 784px.

**Multilayer Perceptron**

```python
# Hidden layer with RELU activation
layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']),\
    biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)
# Output layer with linear activation
logits = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
```

You've seen the linear function `tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])` before, also known as `xw + b`. Combining linear functions together using a ReLU will give you a two layer network.

**Optimizer**

```python
# Define loss and optimizer
cost = tf.reduce_mean(\
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)
```

This is the same optimization technique used in the Intro to TensorFLow lab.

**Session**

```python
# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
```

The MNIST library in TensorFlow provides the ability to receive the dataset in batches. Calling the `mnist.train.next_batch()` function returns a subset of the training data.



### 6.3 Save and Restore TensorFlow Models

Training a model can take hours. But once you close your TensorFlow session, you lose all the trained weights and biases. If you were to reuse the model in the future, you would have to train it all over again!

Fortunately, TensorFlow gives you the ability to save your progress using a class called [`tf.train.Saver`](https://www.tensorflow.org/api_docs/python/tf/train/Saver). This class provides the functionality to save any [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) to your file system.

#### Saving Variables

Let's start with a simple example of saving `weights` and `bias` Tensors. For the first example you'll just save two variables. Later examples will save all the weights in a practical model.

```python
import tensorflow as tf

# The file path to save the data
save_file = './model.ckpt'

# Two Tensor Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()

with tf.Session() as sess:
    # Initialize all the Variables
    sess.run(tf.global_variables_initializer())

    # Show the values of weights and bias
    print('Weights:')
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(bias))

    # Save the model
    saver.save(sess, save_file)
```

- The Tensors `weights` and `bias` are set to random values using the [`tf.truncated_normal()`](https://www.tensorflow.org/api_docs/python/tf/truncated_normal) function. The values are then saved to the `save_file` location, "model.ckpt", using the [`tf.train.Saver.save()`](https://www.tensorflow.org/api_docs/python/tf/train/Saver#save) function. (The ".ckpt" extension stands for "checkpoint".)

- If you're using TensorFlow 0.11.0RC1 or newer, a file called "model.ckpt.meta" will also be created. **This file contains the TensorFlow graph.**

#### Loading Variables

Now that the Tensor Variables are saved, let's load them back into a new model.

```python
# Remove the previous weights and bias
tf.reset_default_graph()

# Two Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()

with tf.Session() as sess:
    # Load the weights and bias
    saver.restore(sess, save_file)

    # Show the values of weights and bias
    print('Weight:')
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(bias))
```

- You'll notice you still need to create the `weights` and `bias` Tensors in Python. The [`tf.train.Saver.restore()`](https://www.tensorflow.org/api_docs/python/tf/train/Saver#restore) function loads the saved data into `weights` and `bias`.

- Since [`tf.train.Saver.restore()`](https://www.tensorflow.org/api_docs/python/tf/train/Saver#restore) sets all the TensorFlow Variables, you don't need to call [`tf.global_variables_initializer()`](https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer).

#### Save a Trained Model

Let's see how to train a model and save its weights. First start with a model:

```python
# Remove previous Tensors and Operations
tf.reset_default_graph()

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

learning_rate = 0.001
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('.', one_hot=True)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
cost = tf.reduce_mean(\
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

Let's train that model, then save the weights:

```python
import math

save_file = './train_model.ckpt'
batch_size = 128
n_epochs = 100

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(n_epochs):
        total_batch = math.ceil(mnist.train.num_examples / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_features, batch_labels = mnist.train.next_batch(batch_size)
            sess.run(
                optimizer,
                feed_dict={features: batch_features, labels: batch_labels})

        # Print status for every 10 epochs
        if epoch % 10 == 0:
            valid_accuracy = sess.run(
                accuracy,
                feed_dict={
                    features: mnist.validation.images,
                    labels: mnist.validation.labels})
            print('Epoch {:<3} - Validation Accuracy: {}'.format(
                epoch,
                valid_accuracy))

    # Save the model
    saver.save(sess, save_file)
    print('Trained Model Saved.')
```

#### Load a Trained Model

Let's load the weights and bias from memory, then check the test accuracy.

```python
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, save_file)

    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: mnist.test.images, labels: mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))
```

> Test Accuracy: 0.7229999899864197

That's it! You now know how to save and load a trained model in TensorFlow. Let's look at loading weights and biases into modified models in the next section.

### 6.4 Finetuning

#### Loading the Weights and Biases into a New Model

Sometimes you might want to adjust, or "finetune" a model that you have already trained and saved.

However, loading saved Variables directly into a modified model can generate errors. Let's go over how to avoid these problems.

**Naming Error**

TensorFlow uses a string identifier for Tensors and Operations called `name`. If a name is not given, TensorFlow will create one automatically. TensorFlow will give the first node the name `<Type>`, and then give the name `<Type>_<number>` for the subsequent nodes. Let's see how this can affect loading a model with a different order of `weights` and `bias`:

```python
import tensorflow as tf

# Remove the previous weights and bias
tf.reset_default_graph()

save_file = 'model.ckpt'

# Two Tensor Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

saver = tf.train.Saver()

# Print the name of Weights and Bias
print('Save Weights: {}'.format(weights.name))
print('Save Bias: {}'.format(bias.name))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, save_file)

# Remove the previous weights and bias
tf.reset_default_graph()

# Two Variables: weights and bias
bias = tf.Variable(tf.truncated_normal([3]))
weights = tf.Variable(tf.truncated_normal([2, 3]))

saver = tf.train.Saver()

# Print the name of Weights and Bias
print('Load Weights: {}'.format(weights.name))
print('Load Bias: {}'.format(bias.name))

with tf.Session() as sess:
    # Load the weights and bias - ERROR
    saver.restore(sess, save_file)
```

The code above prints out the following:

> Save Weights: Variable:0
>
> Save Bias: Variable_1:0
>
> Load Weights: Variable_1:0
>
> Load Bias: Variable:0
>
> ...
>
> InvalidArgumentError (see above for traceback): Assign requires shapes of both tensors to match.
>
> ...

You'll notice that the `name` properties for `weights` and `bias` are different than when you saved the model. This is why the code produces the "Assign requires shapes of both tensors to match" error. The code `saver.restore(sess, save_file)` is trying to load weight data into `bias` and bias data into `weights`.

Instead of letting TensorFlow set the `name` property, **let's set it manually:**

```python
import tensorflow as tf

tf.reset_default_graph()

save_file = 'model.ckpt'

# Two Tensor Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]), name='weights_0')
bias = tf.Variable(tf.truncated_normal([3]), name='bias_0')

saver = tf.train.Saver()

# Print the name of Weights and Bias
print('Save Weights: {}'.format(weights.name))
print('Save Bias: {}'.format(bias.name))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, save_file)

# Remove the previous weights and bias
tf.reset_default_graph()

# Two Variables: weights and bias
bias = tf.Variable(tf.truncated_normal([3]), name='bias_0')
weights = tf.Variable(tf.truncated_normal([2, 3]) ,name='weights_0')

saver = tf.train.Saver()

# Print the name of Weights and Bias
print('Load Weights: {}'.format(weights.name))
print('Load Bias: {}'.format(bias.name))

with tf.Session() as sess:
    # Load the weights and bias - No Error
    saver.restore(sess, save_file)

print('Loaded Weights and Bias successfully.')
```



> Save Weights: weights_0:0
>
> Save Bias: bias_0:0
>
> Load Weights: weights_0:0
>
> Load Bias: bias_0:0
>
> Loaded Weights and Bias successfully.

That worked! The Tensor names match and the data loaded correctly.

### 6.5 Prevent Over fitting

**Early Stopping**

The first way we **prevent over fitting** is by looking at the performance on our validation set. And stopping to train, as soon as we stop improving. It’s called **early termination.** Another way is to apply regularization.

**Regularization**

- Regularizing means applying artificial constraints on your network, that implicitly reduce
  the number of free parameters. While not making it more difficult to optimize.
- The idea is to add another term to the loss, which penalizes large weights. It's typically achieved by adding the L2 norm of your weights to the loss, multiplied by a small constant.

**Dropout**



## 7. Convolutional Neural Networks

### 7.1 Introduction to CNN

- CovNets are neural networks that share their parameters across space.
- The first step for a CNN is to break up the image into smaller pieces. We do this by selecting a width and height that defines a filter.
- What's important here is that we are **grouping together adjacent pixels** and treating them as a collective. By taking advantage of this local structure, our CNN learns to classify local patterns, like shapes and objects, in an image.
- Having multiple neurons for a given patch ensures that our CNN can learn to capture whatever characteristics the CNN learns are important.
- Remember that the CNN isn't "programmed" to look for certain characteristics. Rather, it learns **on its own** which characteristics to notice.

#### Parameter Sharing

- The weights, `w`, are shared across patches for a given layer in a CNN to detect the cat above regardless of where in the image it is located.
-  Note that as we increase the depth of our filter, the number of weights and biases we have to learn still increases, as the weights aren't shared across the output channels.

- There’s an additional benefit to sharing our parameters. If we did not reuse the same weights across all patches, we would have to learn new parameters for every single patch and hidden layer neuron pair. This does not scale well, especially for **higher fidelity images.** Thus, sharing parameters not only helps us with translation invariance, but also gives us a smaller, more scalable model.



#### Padding

As we can see, the width and height of each subsequent layer decreases in the above scheme.

One way is to simply add a border of `0`s to our original `5x5` image. 



#### Dimensionality

From what we've learned so far, how can we calculate the number of neurons of each layer in our CNN?

Given:

- our input layer has a width of `W` and a height of `H`
- our convolutional layer has a filter size `F`
- we have a stride of `S`
- a padding of `P`
- and the number of filters `K`,

the following formula gives us the width of the next layer: `W_out =[ (W−F+2P)/S] + 1`.

The output height would be `H_out = [(H-F+2P)/S] + 1`.

And the output depth would be equal to the number of filters `D_out = K`.

The output volume would be `W_out * H_out * D_out`.

Knowing the dimensionality of each additional layer helps us understand how large our model is and how our decisions around filter size and stride affect the size of our network.

**Code Example**

```python
input = tf.placeholder(tf.float32, (None, 32, 32, 3))
filter_weights = tf.Variable(tf.truncated_normal((8, 8, 3, 20))) # (height, width, input_depth, output_depth)
filter_bias = tf.Variable(tf.zeros(20))
strides = [1, 2, 2, 1] # (batch, height, width, depth)
padding = 'SAME'
conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias
```

Note the output shape of `conv` will be [1, 16, 16, 20]. It's 4D to account for batch size, but more importantly, it's not [1, 14, 14, 20]. This is because the padding algorithm TensorFlow uses is not exactly the same as the one above. An alternative algorithm is to switch `padding` from `'SAME'` to `'VALID'` which would result in an output shape of [1, 13, 13, 20]. If you're curious how padding works in TensorFlow, read [this document](https://www.tensorflow.org/api_guides/python/nn#Convolution).

In summary TensorFlow uses the following equation for 'SAME' vs 'VALID'

**SAME Padding**, the output height and width are computed as:

`out_height` = ceil(float(in_height) / float(strides[1]))

`out_width` = ceil(float(in_width) / float(strides[2]))

**VALID Padding**, the output height and width are computed as:

`out_height` = ceil(float(in_height - filter_height + 1) / float(strides[1]))

`out_width` = ceil(float(in_width - filter_width + 1) / float(strides[2]))



### 7.2 Visualizing CNNs

The CNN we will look at is trained on ImageNet as described in [this paper](http://www.matthewzeiler.com/pubs/arxive2013/eccv2014.pdf) by Zeiler and Fergus. In the images below (from the same paper), we’ll see *what* each layer in this network detects and see *how* each layer detects more and more complex ideas.

**Layer 1**

<img src=assets/7_2_1.png width=100 >

Example patterns that cause activations in the first layer of the network. These range from simple diagonal **lines** (top left) to green **blobs** (bottom middle).

The images above are from Matthew Zeiler and Rob Fergus' [deep visualization toolbox](https://www.youtube.com/watch?v=ghEmQSxT6tw), which lets us visualize what each layer in a CNN focuses on.

Each image in the above grid represents a pattern that causes the neurons in the first layer to activate - in other words, they are patterns that the first layer recognizes. The top left image shows a -45 degree line, while the middle top square shows a +45 degree line. 

**Layer 2**

<img src=assets/7_2_2.png width=400 >

A visualization of the second layer in the CNN. Notice how we are picking up more complex ideas like **circles and stripes**. The gray grid on the left represents how this layer of the CNN activates (or "what it sees") based on the corresponding images from the grid on the right.

**The CNN learns to do this on its own.** There is no special instruction for the CNN to focus on more complex objects in deeper layers. That's just how it normally works out when you feed training data into a CNN.

**Layer 3:**

<img src=assets/7_2_3.png width=500 >

A visualization of the third layer in the CNN. The gray grid on the left represents how this layer of the CNN activates (or "what it sees") based on the corresponding images from the grid on the right.

The third layer picks out complex combinations of features from the second layer. These include things like **grids**, and **honeycombs** (top left), wheels (second row, second column), and even faces (third row, third column).

**Layer 5:**

<img src=assets/7_2_4.png width=300 >



The last layer picks out the highest order ideas that we care about for classification, like dog faces, bird faces, and bicycles.

### 7.3 TF Conv layer

TensorFlow provides the [`tf.nn.conv2d()`](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d) and [`tf.nn.bias_add()`](https://www.tensorflow.org/api_docs/python/tf/nn/bias_add) functions to create your own convolutional layers.

```python
# Output depth
k_output = 64

# Image Properties
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter
filter_size_width = 5
filter_size_height = 5

# Input/Image
input = tf.placeholder(
    tf.float32,
    shape=[None, image_height, image_width, color_channels])

# Weight and bias
weight = tf.Variable(tf.truncated_normal(
    [filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# Apply Convolution
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
# Add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# Apply activation function
conv_layer = tf.nn.relu(conv_layer)
```

The code above uses the [`tf.nn.conv2d()`](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d) function to compute the convolution with `weight` as the filter and `[1, 2, 2, 1]` for the strides. TensorFlow uses a stride for each `input` dimension, `[batch, input_height, input_width, input_channels]`. We are generally always going to set the stride for `batch` and `input_channels` (i.e. the first and fourth element in the `strides` array) to be `1`.

You'll focus on changing `input_height` and `input_width` while setting `batch` and `input_channels` to 1. The `input_height` and `input_width` strides are for striding the filter over `input`. This example code uses a stride of 2 with 5x5 filter over `input`.

The [`tf.nn.bias_add()`](https://www.tensorflow.org/api_docs/python/tf/nn/bias_add) function adds a 1-d bias to the last dimension in a matrix.

### 7.4 TF Max Pooling

The image above is an example of [max pooling](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer) with a 2x2 filter and stride of 2. The four 2x2 colors represent each time the filter was applied to find the maximum value.

For example, `[[1, 0], [4, 6]]` becomes `6`, because `6` is the maximum value in this set. Similarly, `[[2, 3], [6, 8]]` becomes `8`.

Conceptually, the benefit of the max pooling operation is to reduce the size of the input, and allow the neural network to focus on only the most important elements. Max pooling does this by only retaining the maximum value for each filtered area, and removing the remaining values.

TensorFlow provides the [`tf.nn.max_pool()`](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool) function to apply [max pooling](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer) to your convolutional layers.

```python
...
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
conv_layer = tf.nn.bias_add(conv_layer, bias)
conv_layer = tf.nn.relu(conv_layer)
# Apply Max Pooling
conv_layer = tf.nn.max_pool(
    conv_layer,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
```

The [`tf.nn.max_pool()`](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool) function performs max pooling with the `ksize` parameter as the size of the filter and the `strides` parameter as the length of the stride. 2x2 filters with a stride of 2x2 are common in practice.

The `ksize` and `strides` parameters are structured as 4-element lists, with each element corresponding to a dimension of the input tensor (`[batch, height, width, channels]`). For both `ksize` and `strides`, the batch and channel dimensions are typically set to `1`.



## 8. [Project] Traffic Sign Classifier

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. Specifically, you'll train a model to classify traffic signs from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

### Set Up Your Environment

**CarND Starter Kit**

Install the car nanodegree starter kit if you have not already done so: [carnd starter kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

**TensorFlow**

If you have access to a GPU, you should follow the TensorFlow instructions for [installing TensorFlow with GPU support](https://www.tensorflow.org/get_started/os_setup#optional_install_cuda_gpus_on_linux).

Once you've installed all of the necessary dependencies, you can install the `tensorflow-gpu` package:

```
pip install tensorflow-gpu
```

**Amazon Web Services**

Instead of a local GPU, you could use Amazon Web Services to launch an EC2 GPU instance. (This costs money.)

1. [Follow the Udacity instructions](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/614d4728-0fad-4c9d-a6c3-23227aef8f66/concepts/f6fccba8-0009-4d05-9356-fae428b6efb4) to launch an EC2 GPU instance with the `udacity-carnd` AMI.
2. Complete the **Setup** instructions.

### Start the Project

1. [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32.

2. Clone the project and start the notebook.

   ```
   git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
   cd CarND-Traffic-Sign-Classifier-Project
   ```

3. Launch the Jupyter notebook: `jupyter notebook Traffic_Sign_Classifier.ipynb`

4. Check out the [project rubric](https://review.udacity.com/#!/rubrics/481/view)

5. Follow the instructions in the notebook

6. Write your project report

### Submission

Before submitting, make sure your project covers all of the rubric points, which can be found [here](https://review.udacity.com/#!/rubrics/481/view).

When you are ready to submit your project, collect the following files and compress them into a single archive for upload. Alternatively, upload your files to github to link to the project repository:

- The Traffic_Sign_Classifier.ipynb notebook file with all questions answered and all code cells executed and displaying output.
- An HTML or PDF export of the project notebook with the name report.html or report.pdf.
- Any additional datasets or images used for the project that are not from the German Traffic Sign Dataset. ***Please do not include the project data set provided in the `traffic-sign-data.zip` file.\***
- Your writeup report as a markdown or pdf file

If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well.

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.



## 9. Keras

[Keras](https://keras.io/) makes coding deep neural networks simpler. To demonstrate just how easy it is, you're going to build a simple fully-connected network in a few dozen lines of code.

We’ll be connecting the concepts that you’ve learned in the previous lessons to the methods that Keras provides.

The network you will build is similar to Keras’s [sample network](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py) that builds out a convolutional neural network for [MNIST](http://yann.lecun.com/exdb/mnist/). However for the network you will build you're going to use a small subset of the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) dataset that you've used previously.

The general idea for this example is that you'll first load the data, then define the network, and then finally train the network.

### 9.1 Neural Networks in Keras

Here are some core concepts you need to know for working with Keras.

#### Sequential Model

```python
from keras.models import Sequential

#Create the Sequential model
model = Sequential()
```

The [keras.models.Sequential](https://keras.io/models/sequential/) class is a wrapper for the neural network model. It provides common functions like `fit()`, `evaluate()`, and `compile()`. We'll cover these functions as we get to them. Let's start looking at the layers of the model.

#### Layers

A Keras layer is just like a neural network layer. There are fully connected layers, max pool layers, and activation layers. You can add a layer to the model using the model's `add()` function. For example, a simple model would look like this:

```python
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

#Create the Sequential model
model = Sequential()

#1st Layer - Add a flatten layer
model.add(Flatten(input_shape=(32, 32, 3)))

#2nd Layer - Add a fully connected layer
model.add(Dense(100))

#3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))

#4th Layer - Add a fully connected layer
model.add(Dense(60))

#5th Layer - Add a ReLU activation layer
model.add(Activation('relu'))
```

Keras will automatically infer the shape of all layers after the first layer. This means you only have to set the input dimensions for the first layer.

The first layer from above, `model.add(Flatten(input_shape=(32, 32, 3)))`, sets the input dimension to (32, 32, 3) and output dimension to (3072=32 x 32 x 3). The second layer takes in the output of the first layer and sets the output dimensions to (100). This chain of passing output to the next layer continues until the last layer, which is the output of the model.

#### Code

In this quiz you will build a multi-layer feedforward neural network to classify traffic sign images using Keras.

To get started, review the Keras documentation about models and layers. The Keras example of a [Multi-Layer Perceptron](https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py) network is similar to what you need to do here. Use that as a guide, but keep in mind that there are a number of differences.

- **See `Scripts\09.Keras\5_MLP` for more details.**

**Data Download**

The data set used in these quizzes can be downloaded [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/March/58dbf6d5_small-traffic-set/small-traffic-set.zip).



#### Convolutions

1. Build from the previous network.
2. Add a [convolutional layer](https://keras.io/layers/convolutional/#convolution2d) with 32 filters, a 3x3 kernel, and valid padding before the flatten layer.
3. Add a ReLU activation after the convolutional layer.
4. Train for 3 epochs again, should be able to get over 50% accuracy.

Hint 1: The Keras example of a [convolutional neural](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py) network for MNIST would be a good example to review.

- **See `Scripts\09.Keras\06_conv` for more details.**

#### Pooling

1. Build from the previous network
2. Add a 2x2 [max pooling layer](https://keras.io/layers/pooling/#maxpooling2d) immediately following your convolutional layer.
3. Train for 3 epochs again. You should be able to get over 50% training accuracy.

#### Dropout

1. Build from the previous network.
2. Add a [dropout](https://keras.io/layers/core/#dropout) layer after the pooling layer. Set the dropout rate to 50%.

#### Test

Once you've picked out your best model, it's time to test it!

1. Try to get the highest validation accuracy possible. Feel free to use all the previous concepts and train for as many epochs as needed.
2. Select your best model and train it one more time.
3. Use the test data and the [Keras `evaluate()`](https://keras.io/models/model/#evaluate) method to see how well the model does.