# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/blanklist/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images.
* The size of the validation set is 4410 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It outputs an image from each image class. This confirms the sort of images which can be expected among the data set.

![data visualization image](https://github.com/blanklist/CarND-Traffic-Sign-Classifier/blob/master/data_visualization.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color of an image does not assist neural network classifiers.

Here is an example of a traffic sign image before and after grayscaling.

![rgb to gray](https://github.com/blanklist/CarND-Traffic-Sign-Classifier/blob/master/rbg_to_gray.png)

As a last step, I normalized the image data because this helps the image's data distribution remain consistent across the data. Convergence occurs faster while training the network.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was a slight variation on LeNet. That variation had to do with implementation so, the layers remain the same:

![LeNet diagram](https://github.com/blanklist/CarND-Traffic-Sign-Classifier/blob/master/mylenet.png)
(image found here: deeplearning.net/tutorial/lenet.html)

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5X5     	| 1x1 stride, outputs 28x28x6|
| RELU					|												|
| Average pooling	      	| 2x2 stride, outputs 14x14x6 				|
| Convolution 5x5       | 1x1 stride, outputs 10x10x16|      									|
| RELU                                  |
| Average pooling  		| 2x2 stride, outputs 5x5x16        									|
| Fully Connected Convolution				| outputs 48120        									|
| Fully Connected Convolution				| outputs 10164												|
| Fully Connected Convolution				| final output												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer. 
Batch size was set to 200 images. 
Epochs were set to 20. 
The learning rate was set to 0.006.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.1%
* validation set accuracy of 93.5% 
* test set accuracy of 90.8%

If a well known architecture was chosen:
* What architecture was chosen?
I chose the LeNet architecture as it is well documented and seemed a good choice that I could then make small adjustments to improve upon.

* Why did you believe it would be relevant to the traffic sign application?
As this was my first exposure to convolutional neural networks, my initial concern was creating a functional model. LeNet is a well known and well documented model which was referenced in the lecture material. Exterior obligations (a career) and deadlines limited my ability to adjust and explore different techniques to achieve improvements in model accuracy. Next steps for this model would be to add dropout functionality and experiment. Beyond that I would consider approaching the project with an entirely new model. I would also spend some time automating the calculations for image size for input and output of layers.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![five images](https://github.com/blanklist/CarND-Traffic-Sign-Classifier/blob/master/five_images.png)

The first and second images might be difficult to classify because of the amount of blur making the lines and reultant shapes or numbers indistinct. The third image is very clear and simplistic. The fourth and fifth images have clarity though have many features inside of their respective circles and behind the diagonal line.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100 km/h      		| 100 km/h   									| 
| No passing for [trucks]       | No passing for [trucks] 										|
| End of all speed and passing limits| End of all speed and passing limits											|
| End of no passing by [trucks]	| End of no passing by [trucks]					 				|
| End of speed limit (80km/h)	| End of speed limit (80km/h)      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 91%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 31st cell of the Ipython notebook.

For the first image, the model is nearly sure that this is a 100 km/h sign (probability of 0.99), and the image does contain a 100 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (100km/h)   									| 
| .00     				| Speed limit (30km/h) 										|
| .00					| Speed limit (120km/h)											|
| .00	      			| Speed limit (80km/h)					 				|
| .00				    | Roundabout mandatory      							|


For the second image, the model is completely sure that this is a no passing for [trucks] sign (probability of 1), and the image does contain a no passing for [trucks] sign.  The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| no passing for [trucks]   									| 
| .00     				| Slippery road 										|
| .00					| No passing											|
| .00	      			| Speed limit (60km/h)					 				|
| .00				    | Speed limit (100km/h)      							|


For the third image, the model is completely sure that this is a end of all speed and passing limits sign (probability of 1.0), and the image does contain a End of all speed and passing limits sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| End of all speed and passing limits   									| 
| .00     				| End of no passing 										|
| .00					| End of speed limit (80km/h)											|
| .00	      			| Keep right					 				|
| .00				    | Dangerous curve to the right      							|


For the fourth image, the model is nearly sure that this is a End of no passing by [trucks] sign (probability of 0.99), and the image does contain a End of no passing by [trucks] sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| End of no passing by [trucks]   									| 
| .00     				| End of speed limit (80km/h) 										|
| .00					| Speed limit (30km/h)											|
| .00	      			| End of no passing					 				|
| .00			    | Speed limit (100km/h)      							|


For the fifth image, the model is nearly sure that this is a End of speed limit (80km/h) sign (probability of 0.99), and the image does contain a End of speed limit (80km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| End of speed limit (80km/h)   									| 
| .00     				| End of no passing by [trucks] 										|
| .00					| End of no passing											|
| .00	      			| End of all speed and passing limits					 				|
| .00				    | Speed limit (80km/h)      							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


