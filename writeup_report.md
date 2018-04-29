# **Behavioral Cloning** 

## Writeup report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pics/center_2016_12_01_13_40_11_279.jpg "Random Pic. Train Data 1"
[image2]: ./pics/center_2016_12_01_13_42_42_686.jpg "Random Pic. Train Data 2"
[image3]: ./pics/right_2016_12_01_13_32_52_652.jpg "Random Pic. Train Data 3"
[image4]: ./pics/conv_1.png "Conv. Layer 1"
[image5]: ./pics/conv_2.png "Conv. Layer 2"
[image6]: ./pics/conv_3.png "Conv. Layer 3"
[image7]: ./pics/conv_4.png "Conv. Layer 4"
[image8]: ./pics/conv_5.png "Conv. Layer 5"
[image9]: ./pics/nvidia_model.png "Nvidia Model Architecture"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5.zip containing a trained convolution neural network 
* writeup_report.md  summarizing the results
* visualization.py for generation figures automatically

Notes:
* I updated Keras to Version 2.0.3 (via pip)
* I installed pydot (pip install pydot) and installed graphviz for keras model visualization
* For CNN layer visulization I used keras-toolbox (pip install keras-toolbox)
* I had to zip model.h5 cause of githubs 100MB limit. Please unzip in manually!

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of various convolution neural network layers and followed by flully connceted layer.  For details about the implemented model architecture please see section 2.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 83). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 89, 96, 98, 100, 102). 

The model was trained and validated on to ensure that the model was not overfitting (code line 109). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 105).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data and wich one I used, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to implement the architecture shown and described in more detail at https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf .

My first step was to use a simple fully connected neural network. Mainly to test my own setup and test runs. Afterwards I implemented the NVIDIA architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

The model was gradually improved by introducing the dropout mechanism (for CNNs the so-called SpatialDropout as described in more detail at https://www.tensorflow.org/api_docs/python/tf/keras/layers/SpatialDropout2D) and adjusting the factor. Especially with SpatialDropout I noticed that a factor >0.5 is not recommended.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Especially at places where a road boundary was only dirt or where the curves were very sharp. Increasing the epochs has also had a positive (albeit small) effect here.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 77-103) consisted of various convolution neural networks and fully connected networks with the following layers and layer sizes (the visualization was generated automatically (visualization.py lines 54-55):

![alt text][image9]

To better understand the principle of CNNs and their learned filters, I set myself the task of visualizing them. This could be very helpful for future projects and tasks. The visualization is therefore also outsourced to the visualization.py module. I used the recommended keras-toolbox (https://github.com/hadim/keras-toolbox).

For the following three images, the individual feature sets are now visualized:
![alt text][image2] ![alt text][image3] ![alt text][image1]

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

It is impressive to see how the network learns its filter for the detection of road boundaries and edges in these layers. The ability to abstract features mentioned in the corresponding Udacity lessons can also be observed (only few white vertical lines in the last layer compared to the first layers).

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track one using center lane driving. 
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. In the end, however, I used the data provided by udacity. However, my data was very revealing for initial tests.

I deliberately didn't consider the second track for the training. As a final test for my then hopefully generalized model (the model shows an acceptable behavior here).

To augment the data sat, I also flipped images and angles thinking to have more training data. I also added to this augmented data a correction factor of +/-0.25. So I interpret the data as a new independent lap of the car.

After the collection process, I had almost 39.000 number of data points. I then preprocessed this data by normalization and cropping.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by that the car can complete a full lap without violating any of the required requirements. I used an adam optimizer so that manually training the learning rate wasn't necessary.
