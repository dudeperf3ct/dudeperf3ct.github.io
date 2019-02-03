---
layout:     post
title:      Mystery of Object Detection
date:       2019-01-07 12:00:00
summary:    This post will provide an brief introduction to different architecture in object detection.
categories: object detection
published : false
---


# Object Detection

In this notebook, 


> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Style%20Transfer/style_transfer_keras.ipynb), [PyTorch](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Style%20Transfer/style_transfer_pytorch.ipynb) and [fastai](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Style%20Transfer/style_transfer_fastai.ipynb).  

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.

-meme

Feel free to jump anywhere,
- [Loss Functions](#loss-functions)
  - [Classification Loss](#classification-loss)
  - [Regression Loss](#regression-loss)
- [Introduction to Object Detection](#introduction-to-object-detection)
  - [Viola-Jones](#viola-jones)
  - [OverFeat](#overfeat)
  - [R-CNN](#r-cnn)
  - [Fast R-CNN](#fast-r-cnn)
  - [Faster R-CNN](#faster-r-cnn)
  - [R-FCN](#r-fnn)
  - [SSD](#ssd)
  - [YOLO](#yolo)
  - [RetinaNet](#retinanet)
  - [Backbones](#backbones)
    - [MobileNet](#mobilenet)
    - [FPN](#fpn)
    - [ResNet](#resnet)
    - [ResNext](#resnext)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)


# Loss Functions

Loss functions are the heart of deep learning algorithms (*in case you are wondering, backprop the soul*). Loss functions tells the model how good the model is at particular task. Depending on the problem to solve, almost all model aim to minimize the loss. Also, did you notice one thing in particular about loss functions and non-linear functions, they are all "differentiable functions". Yes, we may also call deep learning as "differentiable programming". As there is "No Free Lunch" theorem in machine learning, which states that no one particular model can solve all the problems. Similarly, there is also no one particular loss function which when minimized(or maximize) will solve any task. If we make any changes to our model in hope(trying different hyperparameters) of creating better model, loss function will tell if we’re getting better model than previous model trained. If predictions of the model are totally off, loss function will output a higher number. If they’re pretty good, it’ll output a lower number. Designing loss functions to solve our particular task is one of the critical steps in deep learning, if we choose a poor error(loss) function and obtain unsatisfactory results, the fault is ours for badly specifying the goal of the search.

Loss function is defined in [Deep Learning book](https://www.deeplearningbook.org/contents/ml.html) as, 

> The function we want to minimize or maximize is called the objective function or criterion. When we are minimizing it, we may also call it the cost function, loss function, or error function.

There are lots many loss functions. But, broadly we can classify loss functions into two categories.

**Classification Loss**

As the name suggests, this loss will help with any task which requires classification. We are given k categories and our job is to make sure our model is good job in classifying x number of examples in k categories. An example is, we are given 1.2 million images of 1000 different categories, and our task it to classify each given image into it's 1000 categories.  

Cross Entropy Loss

Cross-entropy loss is often simply referred to as “cross-entropy,” “logarithmic loss,” “logistic loss,” or “log loss” for short. 
There are two interpretation of cross entropy. One through information theory and other through probabilistic view. 

Information theory view

The entropy rate of a data source means the average number of bits per symbol needed to encode it without any loss of information. Entropy of probability distribution p is given by $$H(p)  = -\sum_{i}^{}p(i)\log_{2}{p(i)}$$. Let p be the true distrubtion and q be the predicted distribution over our labels, then cross entropy of both distribution is defined as. $$H(p, q)  = -\sum_{i}^{}p(i)\log_{2}{q(i)}$$. It looks like pretty similar to equation of entropy above but instead of computing log of true probability, we compute log of predicted probability distribution.

The cross-entropy compares the model’s prediction with the label which is the true probability distribution. Cross entropy will grow large if predicted probability for true class is close to zero. But it goes down as the prediction gets more and more accurate. It becomes zero if the prediction is perfect i.e. our predicted distribution is equal to true distribution. KL Divergence(relative entropy) is the extra bit which exceeds if we remove entropy from cross entropy.


Aurélien Géron explains amazingly how entropy, cross entropy and KL Divergence pieces are connected in this [video](https://www.youtube.com/watch?v=ErfnhcEV1O8).

Probabilistic View

The output obtained from last softmax(or sigmoid for binary class) layer of the model can be interpreted as normalized class probabilities and we are therefore minimizing the negative log likelihood of the correct class or we are performing Maximum Likelihood Estimation (MLE). 

As David Silver would like to say, let's make it concrete with an example. For example consider we get an output of [0.1, 0.5, 0.4] (cat, dog, mouse) where the actual or expected output is [1, 0, 0] i.e. it is a cat. But our model predicted that given input has only 10% probability of being a cat, 50% probability of being dog and 40% of chance being a mouse. This being a multi-class classification, we can calculate the cross entropy using the formula for $$\mathbf{L_{mce}}$$ below. 

Another example for binary class can be as follows. The models outputs [0.4, 0.6] (cat, dog) whereas the input image is a cat i.e. actual output is [1, 0]. Now, we can use $$\mathbf{L_{bce}}$$ from below to calculate the loss and backpropgate the error and tell the model to correct its weight so as to get the output correct next time.

There are two different types of cross entropy functions depending on number of classes to classify into.

- Binary Classification

As name suggests, there will be binary(two) classes. If we have two classes to classify our images into, then we use binary cross entropy. Cross entropy loss penalizes heavily the predictions that are confident but wrong. Suppose, $$\mathbf{y\hat}$$ is our predicted output by the model and $$\mathbf{y}$$ is target(actual or expected) value. For M example, binary cross entropy can be forumlated as, 

$$
\begin{aligned}
\mathbf{L_{bce}} = - \frac{1}{M}\sum_{i=1}^{M}(\mathbf{y_{i}}\log_{}{\mathbf{\hat{y}_{i}}} + (1-\mathbf{y}_{i})\log_{}{(1-\mathbf{\hat{y}_{i}})})
\end{aligned}
$$


- Multi-class Classification

As name suggests, if there are more than two classes that we want our images to be classified into, then we use multi-class classification error function. It is used as a loss function in neural networks which have softmax activations in the output layer. The model outputs the probability the example belonging to each class. For classifying into C classes, where C > 2, multi-class classification is given by,  

$$
\begin{aligned}
\mathbf{L_{mce}} = - \sum_{c=1}^{C}(\mathbf{y_c}\ln{\mathbf{\hat{y}_c}})
\end{aligned}
$$


**Regression Loss**

In regression, model outputs a number. This number is then compared with our expected value to get a measure of error. For example, we wanted to predict the prices of houses in the neighbourhood. So, we give our model different features(like number of bedrooms, number of bathrooms, area, etc) and ask the model to output the price of house.

- Mean Squared Error(MSE)

These error functions are easy to define. As the name suggests, we are taking square of error and then mean of these sqaured error functions. It’s only concerned with the average magnitude of error irrespective of their direction. However, due to squaring, predictions which are far away from actual values are penalized heavily in comparison to less deviated predictions. Suppose, $$\mathbf{y\hat}$$ is our predicted output by the model and $$\mathbf{y}$$ is target(actual or expected) value. For M training example, mse loss can be forumlated as, 

$$
\begin{aligned}
\mathbf{L_{mse}} = \frac{1}{M}\sum_{i=0}^{M} (\mathbf{y_{i}} - \mathbf{\hat{y}_{i}})^2
\end{aligned}
$$

- Mean Absolute Error(MAE)

Similar to one above, this loss takes absolute error difference between target and predicted output. Like MSE, this as well measures the magnitude of error without considering their direction. The difference is MAE is more robust to outliers since it does not make use of square.

$$
\begin{aligned}
\mathbf{L_{mae}} = \frac{1}{M}\sum_{i=0}^{M} |\mathbf{y_{i}} - \mathbf{\hat{y}_{i}}|
\end{aligned}
$$

- Root Mean Squared Error(RMSE)

Root mean square error will be just taking root of above \mathbf{L_{mse}}. The MSE penalizes large errors more strongly and therefore is very sensitive to outliers. To avoid this, we usually use the squared root version.

$$
\begin{aligned}
\mathbf{L_{rmse}} = \sqrt{\frac{1}{M}\sum_{i=0}^{M} (\mathbf{y_{i}} - \mathbf{\hat{y}_{i}})^2}
\end{aligned}
$$

There are also other loss functions like Focal Loss(which we define in RetinaNet), SVM Loss(Hinge), KL Divergence, Huber Loss etc.

*In next post, we will discuss some popular loss functions and where are they used. Stay tuned!*

# Introduction to Object Detection

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' />
</p>


<span class='red'>I-know-everything:</span> 

<span class='green'>I-know-nothing:</span> 

<span class='red'>I-know-everything:</span> 


Now, that you have understood what we are doing in object detection. Let's look at some of the algorithms we can use to create such cool object detectors. *I mean very cool.*

## Viola Jones Detector

In early 2000s, deep learning where in their infancy or deep learning where not everything, in [this paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) Viola and Jones proposed a machine learning approach for visual object detection which is capable of processing images extremely rapidly and achieving high detection rates. The algorithm consisted of 3 important parts, integral images which consisted of calculating about 60,,000 image features, AdaBoost classifier, a boosting process which is collection of weak classifier and cascading, where more complex classifier are cascaded in a chain. Intutivley, they created a chain of suppose 3 classifier, where each sub-window (field of view) classifies it as a "face or not a face". Those sub-windows which are not intially rejected get passed on to next classifier and so no. If any classifier rejects sub-window, no further processing is involved. So, it was a degenerate decision classifier. This process allowed quick selection of faces and discarding backgrounds very quickly. 

For example, in below classifier, 1 feature classifier achieves 100% detection rate with 50% false positive rate, 2 feature classifier with 100% detection rate and 40% false positive rate(20% cumulative) and 20 feature classifier achieve 100% detection rate with 10% false positive rate(2% cumulative).

-viola_jones.png

Here are some results,

-viola_jones_result.png

The real-time detector ran at 15 frames per second on a conventional 700 MHz Intel Pentium III.

For further, take a look at cool explaination by Dr. Mike Pound on [Viola -Jones Algorithm](https://www.youtube.com/watch?v=uEJ71VlUmMQ) on Computerphile.


## OverFeat 

One of the first deep learning approach using ConvNets was developed by LeCunn et al in architecture called [Overfeat](https://arxiv.org/pdf/1312.6229.pdf). They provide integrated approach to object detection, recognition and localization with a single ConvNet. As we have discussed before, in this algorithm there are two parts of network, classification and localization. The classification network(Overfeat architecture) is trained on Imagenet classifying object into one of 1000 categories. The classifier layers of classification network is replaced by regression network which predicts object bounding box at each spatial location and scale. In OverFeat, the region-wise features come from a sliding window of one aspect ratio over a scale pyramid. These features are used to simultaneously determine the location and category of objects. On the 200-class ILSVRC2013 detection dataset, OverFeat achieved mean average precision (mAP) of 24.3%. 

The working of algorithm can be explained by an example shown below.

Using a sliding window approach method, which is effective in ConvNets as they share weights, the algorithm uses 6 different scales of input which are then presented to classifier to predict the class for each window for different resolutions as shown in top left and top right example. The regression then predicts the location scale of object with respect to each window as shown in bottom left and then these bounding boxes are merged and accumulated to a small number of objects as shown in bottom right. The
various aspect ratios of the predicted bounding boxes shows that the network is able to cope with various object poses.



## R-CNN

Introduction for using CNN for object detection gave rise to whole new networks and kept pushing the boundary of state-of-the-art detectors. Quickly after OverFeat, Grishick et al proposed a method where they used selective search to extract 2000 regions which they called "region proposals" (regions with high probability of containing objects). Hence the name, Regions with CNN features, [R-CNN](https://arxiv.org/pdf/1311.2524.pdf). They perform classification and regression on these 2000 region proposals. This result improved the previous result set by Overfeat on ILSVRC2013 detection dataset of 24.3% to 31.4%, an astounding 30% improvement. Let's analyse the steps used in the algorithm:

- Extract possible objects using a region proposal method (the most popular one being Selective Search).
- Extract features from each region using a CNN.
- Classify each region with SVMs.

-rcnn.jpg

### Selective Search

The sliding window based approach used a window (grid of size say 7 x 7) which scans across the whole image and send that to classifier to classify if it is an object or not a object. Then there are various aspect ratio to be considered inside an image as different object can have different sizes. So, classifying for each location becomes extremely slow. But what if somehow someone provided us with 2000 potentially object containing regions regardless of their relative sizes and then our only job is to classify and localize based on these 2000 region proposals.

--selective search

Here come the role of selective search, which use an [hierarchical grouping algorithm](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf#algocf.1) which uses a greedy algorithm to iteratively group regions together. This selective search is used in R-CNN to generate 2000 Region Proposals which are then passed to classifier network. 

The classifier network is AlexNet Network which acts as a feature extractor. For each proposal, a 4096-dimensional vector is computed which are then fed into SVM to classify the presence of the object within that candidate region proposal. This 4096-D vector also fed in a linear regressor to adapt the shapes of the bounding box for a region proposal and thus reduce localization errors.

-rcnn_region_proposal

### Problems in R-CNN

- It takes a lot of time to generate 2000 proposals for each image.(*Can we propose a new algorithm to replace these fixed proposals?*)
- Real time object detection requires 47 seconds (*not cool*).
- The selective search algorithm is a fixed algorithm. Therefore, no learning is happening at that stage. This could lead to the generation of bad candidate region proposals.(*Can we propose a new algorithm which is not fixed?*)
- Training requires multiple stages of processing, where first ConvNet are finetuned to produce 4096-D vector. SVM uses these features to classify and in third stage bounding regressor are learned from feature vectors.(*Could we somehow achieve classification and localization in parallel?*)

### Training

Training routine consists of classifying object into N classes

Typical training routine in all object detection algorithm consists of calculating Intersection Over Union(IOU). We will discuss about it below.

### Intersection Over Union (IOU)



## Fast R-CNN

To overcome shortcomings of R-CNN, Grishick proposes [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)  which employs  several  innovations to improve training and testing speed while also increasing detection accuracy. Let's analyse the steps used in the algorithm:

- An input is entire image and a set of object proposals. 
- The network first processes the whole image with several convolutional (conv) and max pooling layers to produce a conv feature map. 
- For each object proposal a region of interest (RoI) pooling layer extracts a fixed-length feature vector from the feature map. - Each feature vector is fed into a sequence of fully connected (fc) layers that finally branch into two sibling output layers: one that produces softmax probability estimates over K object classes plus a catch-all “background” class and another layer that outputs four real-valued numbers for each of the K object classes. Each set of 4 values encodes refined bounding-box positions for one of the K classes.

-fastrcnn.png

### ROI Pooling

The RoI pooling layer uses max pooling to convert the features inside any valid region of interest into a small feature map with a fixed spatial extent of H × W (e.g. 7 x 7). In example below, with input ROI of 5×7, and output of 2×2, the area for each pooling area is 2×3 or 3×3 after rounding. Region of Interest Pooling allowed for sharing expensive computations and made the model much faster.

-roi_pooling.png


### Advantages over R-CNN

- Higher detection quality (mAP) than R-CNN
- Training is single-stage, using a multi-task loss (no need of multi-stage as seen in RCNN)
- Training can update all network layers (end-to-end)
- Avoid feature caching as SVM is replaced by Softmax, no need to store feature vectors (softmax is better than SVM).

### Problems in Fast R-CNN

- Still requires region proposals from selective search algorithm
- At runtime, the detection network processes images in 0.3s (excluding object proposal time)

## Faster R-CNN

To overcome shortcomings of Fast R-CNN, Grishick(again!) et al proposes faster architecture than previous attempts, hence the name Faster R-CNN. The introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals(*finally*). An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. Let's analyse the steps used in the algorithm:

- In the first step, train the RPN. This network is initialized with an ImageNet-pre-trained model and fine-tuned end-to-end for the region proposal task.
- In  the second  step, train a separate detection network by Fast R-CNN using the proposals generated by the step-1 RPN. This detection network is also initialized by the ImageNet pre-trained model.(At this point the two networks do not share convolutional layers)
- Use Fast R-CNN steps above to classify and localize objects


-fasterrcnn.png


### Region Propsal Network (RPN)

RPN were introduce to replace slow selective search which proposes region proposals with fast neural networks. Here is how RPN works:

- First, the picture goes through conv layers and feature maps are extracted
- Then a sliding window is used in RPN for each location over the feature map
- For each location, k (k=9) anchor boxes are used (3 scales of 128, 256 and 512, and 3 aspect ratios of 1:1, 1:2, 2:1) for generating region proposals
- A cls layer outputs 2k scores whether there is object or not for k boxes
- A reg layer outputs 4k for the coordinates (box center coordinates, width and height) of k boxes
- With a size of W×H feature map, there are WHk anchors in total.


-rpn.png


In Faster R-CNN, the “proposals” are dense sliding windows of 3 scales (128, 256, 512) and 3 aspect ratios (1:1, 1:2, 2:1).

### Region of Interest Pooling (ROI)

After the RPN step, we have a bunch of object proposals with no class assigned to them. Our next problem to solve is how to take these bounding boxes and classify them into our desired categories.

The simplest approach would be to take each proposal, crop it, and then pass it through the pre-trained base network. Then, we can use the extracted features as input for a vanilla image classifier. The main problem is that running the computations for all the 2000 proposals is really inefficient and slow.

Faster R-CNN tries to solve, or at least mitigate, this problem by reusing the existing convolutional feature map. This is done by extracting fixed-sized feature maps for each proposal using region of interest pooling. Fixed size feature maps are needed for the R-CNN in order to classify them into a fixed number of classes.

### Region-based CNN

Region-based convolutional neural network (R-CNN) is the final step in Faster R-CNN’s pipeline. After getting a convolutional feature map from the image, using it to get object proposals with the RPN and finally extracting features for each of those proposals (via RoI Pooling), we finally need to use these features for classification. R-CNN tries to mimic the final stages of classification CNNs where a fully-connected layer is used to output a score for each possible object class.

In Faster R-CNN, the R-CNN takes the feature map for each proposal, flattens it and uses two fully-connected layers of size 4096 with ReLU activation.

Then, it uses two different fully-connected layers for each of the different objects:

- A fully-connected layer with N+1 units where N is the total number of classes and that extra one is for the background class.
- A fully-connected layer with 4N units. We want to have a regression prediction

### Advantages over R-CNN and Fast R-CNN

- Higher detection quality (mAP) than R-CNN and Fast R-CNN
- At runtime, the detection network requires 200ms per image

### Problems in Faster R-CNN

- Two different networks(*Can we combine everything in single network?*)
- Four loss functions to optimize (2 for RPN and 2 for Fast R-CNN)

Results from pretrained model using tensorflow Object Detection API using Faster R-CNN with Resnet pretrained model,




## R-FCN







<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology


Mystery of Object Detection - Object Detection

ConvNets - Convolution Neural Networks

neurons - unit

loss function - cost, error or objective function

---

# Further Reading

[Some Thoughts About The Design Of Loss Functions](https://www.ine.pt/revstat/pdf/rs070102.pdf)

[A More General Robust Loss Function](https://arxiv.org/abs/1701.03077)

[Loss Functions](http://cs231n.github.io/linear-classify/)

[Viola Jones Algorithm paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)

[OverFeat](https://arxiv.org/pdf/1312.6229.pdf)

[R-CNN](https://arxiv.org/abs/1311.2524)

[Selective Search for Object Recognition](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf)

[Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)

[Faster R-CNN]()

---

# Footnotes and Credits


[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)

[RCNN Algorithm](https://arxiv.org/abs/1311.2524)

[Selective Search](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf)

[RCNN illustration](https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9)



---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

