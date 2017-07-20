#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 the video of car driving along the track one.

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The video of car driding along track one is also included.

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The Nvidia convolutional neural network is used here.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. But the correction angle is tuned manually to keep the car on the track.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
80 percent of the udacity captured data is used for training and 20% is used for testing.

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Nvidia neural network. I thought this model might be appropriate because some users mentioned in Udacity forum.

The data is splitted to training and test parts. The training data is selected randomly from the Udacity data. 80 percent of data is used for training and 20 percent is used for testing.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, specially after the bridge. To improve the driving behavior in these cases, I changed the angle correction threshold to keep the car on the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:
I found out from the Udacity forum that addin a dropout layer, improves the model performance.

| Layer                 |     Description                                                       |
|:---------------------:|:---------------------------------------------------------------------:|
| Input                 | 160x320x3 RGB image                                                   |
| Cropping              | Crop top 50 pixels and bottom 20 pixels; output shape = 90x320x3      |
| Normalization         | Each new pixel value = old pixel value/255 - 0.5                      |
| Convolution 5x5       | 5x5 kernel, 2x2 stride, 24 output channels, output shape = 43x158x24  |
| RELU                  |                                                                       |
| Convolution 5x5       | 5x5 kernel, 2x2 stride, 36 output channels, output shape = 20x77x36   |
| RELU                  |                                                                       |
| Convolution 5x5       | 5x5 kernel, 2x2 stride, 48 output channels, output shape = 8x37x48    |
| RELU                  |                                                                       |
| Convolution 5x5       | 3x3 kernel, 1x1 stride, 64 output channels, output shape = 6x35x64    |
| RELU                  |                                                                       |
| Convolution 5x5       | 3x3 kernel, 1x1 stride, 64 output channels, output shape = 4x33x64    |
| RELU                  |                                                                       |
| Flatten               | Input 4x33x64, output 8448                                            |
| Fully connected       | Input 8448, output 100                                                |
| Dropout               | Set units to zero with probability 0.5                                |
| Fully connected       | Input 100, output 50                                                  |
| Fully connected       | Input 50, output 10                                                   |
| Fully connected       | Input 10, output 1 (labels)                                           |

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the data provided by Udacity.
