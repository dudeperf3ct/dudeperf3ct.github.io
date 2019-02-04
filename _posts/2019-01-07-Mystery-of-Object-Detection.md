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

Loss functions are the heart of deep learning algorithms (*in case you are wondering, backprop the soul*). Loss functions tells the model how good the model is at particular task. Depending on the problem to solve, almost all model aim to minimize the loss. Also, did you notice one thing in particular about loss functions and non-linear functions, they are all "differentiable functions". Yes, we may also call deep learning as "differentiable programming". As there is "No Free Lunch" theorem in machine learning, which states that no one particular model can solve all the problems. Similarly, there is also no one particular loss function which when minimized(or maximize) will solve any task. If we make any changes to our model in hope(trying different hyperparameters) of creating better model, loss function will tell if we’re getting better model than previous model trained. If predictions of the model are totally off, loss function will output a higher number. If they’re pretty good, it’ll output a lower number. Designing loss functions to solve our particular task is one of the critical steps in deep learning, if we choose a poor error(loss) function and obtain unsatisfactory results, the fault is ours for badly specifying the goal of the search. (*Choose wisely*)

Loss function is defined in [Deep Learning book](https://www.deeplearningbook.org/contents/ml.html) as, 

> The function we want to minimize or maximize is called the objective function or criterion. When we are minimizing it, we may also call it the cost function, loss function, or error function.

There are lot many loss functions. But, broadly we can classify loss functions into two categories.

**Classification Loss**

As the name suggests, this loss will help with any task which requires classification. We are given k categories and our job is to make sure our model is good job in classifying x number of examples in k categories. An example is, we are given 1.2 million images of 1000 different categories, and our task it to classify each given image into it's 1000 categories.  

- **Cross Entropy Loss**

Cross-entropy loss is often simply referred to as “cross-entropy,” “logarithmic loss,” “logistic loss,” or “log loss” for short. 
There are two interpretation of cross entropy. One through information theory and other through probabilistic view. 

**Information theory view**

The entropy rate of a data source means the average number of bits per symbol needed to encode it without any loss of information. Entropy of probability distribution p is given by $$H(p)  = -\sum_{i}^{}p(i)\log_{2}{p(i)}$$. Let p be the true distrubtion and q be the predicted distribution over our labels, then cross entropy of both distribution is defined as. $$H(p, q)  = -\sum_{i}^{}p(i)\log_{2}{q(i)}$$. It looks like pretty similar to equation of entropy above but instead of computing log of true probability, we compute log of predicted probability distribution.

The cross-entropy compares the model’s prediction with the label which is the true probability distribution. Cross entropy will grow large if predicted probability for true class is close to zero. But it goes down as the prediction gets more and more accurate. It becomes zero if the prediction is perfect i.e. our predicted distribution is equal to true distribution. KL Divergence(relative entropy) is the extra bit which exceeds if we remove entropy from cross entropy.


Aurélien Géron explains amazingly how entropy, cross entropy and KL Divergence pieces are connected in this [video](https://www.youtube.com/watch?v=ErfnhcEV1O8).

**Probabilistic View**

The output obtained from last softmax(or sigmoid for binary class) layer of the model can be interpreted as normalized class probabilities and we are therefore minimizing the negative log likelihood of the correct class or we are performing Maximum Likelihood Estimation (MLE). 

As David Silver would like to say, let's make it concrete with an example. For example consider we get an output of [0.1, 0.5, 0.4] (cat, dog, mouse) where the actual or expected output is [1, 0, 0] i.e. it is a cat. But our model predicted that given input has only 10% probability of being a cat, 50% probability of being dog and 40% of chance being a mouse. This being a multi-class classification, we can calculate the cross entropy using the formula for $$\mathbf{L_{mce}}$$ below. 

Another example for binary class can be as follows. The models outputs [0.4, 0.6] (cat, dog) whereas the input image is a cat i.e. actual output is [1, 0]. Now, we can use $$\mathbf{L_{bce}}$$ from below to calculate the loss and backpropgate the error and tell the model to correct its weight so as to get the output correct next time.

There are two different types of cross entropy functions depending on number of classes to classify into.

- **Binary Classification**

As name suggests, there will be binary(two) classes. If we have two classes to classify our images into, then we use binary cross entropy. Cross entropy loss penalizes heavily the predictions that are confident but wrong. Suppose, $$\mathbf{y\hat}$$ is our predicted output by the model and $$\mathbf{y}$$ is target(actual or expected) value. For M example, binary cross entropy can be forumlated as, 

$$
\begin{aligned}
\mathbf{L_{bce}} = - \frac{1}{M}\sum_{i=1}^{M}(\mathbf{y_{i}}\log_{}{\mathbf{\hat{y}_{i}}} + (1-\mathbf{y}_{i})\log_{}{(1-\mathbf{\hat{y}_{i}})})
\end{aligned}
$$


- **Multi-class Classification**

As name suggests, if there are more than two classes that we want our images to be classified into, then we use multi-class classification error function. It is used as a loss function in neural networks which have softmax activations in the output layer. The model outputs the probability the example belonging to each class. For classifying into C classes, where C > 2, multi-class classification is given by,  

$$
\begin{aligned}
\mathbf{L_{mce}} = - \sum_{c=1}^{C}(\mathbf{y_c}\ln{\mathbf{\hat{y}_c}})
\end{aligned}
$$


**Regression Loss**

In regression, model outputs a number. This number is then compared with our expected value to get a measure of error. For example, we wanted to predict the prices of houses in the neighbourhood. So, we give our model different features(like number of bedrooms, number of bathrooms, area, etc) and ask the model to output the price of house.

- **Mean Squared Error(MSE)**

These error functions are easy to define. As the name suggests, we are taking square of error and then mean of these sqaured error functions. It’s only concerned with the average magnitude of error irrespective of their direction. However, due to squaring, predictions which are far away from actual values are penalized heavily in comparison to less deviated predictions. Suppose, $$\mathbf{y\hat}$$ is our predicted output by the model and $$\mathbf{y}$$ is target(actual or expected) value. For M training example, mse loss can be forumlated as, 

$$
\begin{aligned}
\mathbf{L_{mse}} = \frac{1}{M}\sum_{i=0}^{M} (\mathbf{y_{i}} - \mathbf{\hat{y}_{i}})^2
\end{aligned}
$$

- **Mean Absolute Error(MAE)**

Similar to one above, this loss takes absolute error difference between target and predicted output. Like MSE, this as well measures the magnitude of error without considering their direction. The difference is MAE is more robust to outliers since it does not make use of square.

$$
\begin{aligned}
\mathbf{L_{mae}} = \frac{1}{M}\sum_{i=0}^{M} |\mathbf{y_{i}} - \mathbf{\hat{y}_{i}}|
\end{aligned}
$$

- **Root Mean Squared Error(RMSE)**

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
- A classification layer outputs 2k scores whether there is object or not for k boxes
- A regression layer outputs 4k for the coordinates (box center coordinates, width and height) of k boxes
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
- Fully connected layers increase the parameters in network due to which inference time takes toll.(*Can we get rid of dense layers?*)

Results from pretrained model using tensorflow Object Detection API using Faster R-CNN with Resnet pretrained model,



## R-FCN

[R-FCN](https://arxiv.org/pdf/1605.06409.pdf) uses region-based, fully convolutional networks based approach for object detection where almost all computation shared on the entire image. He et al propose a solution of using "position-sensitive score maps" which takes into account both translation invariance for image classification (wherever the object is in image) and translation variance for drawing boxes around the classified object i.e. object detection. Essentially, these score maps are convolutional feature maps that have been trained to recognize certain parts of each object. Let's analyse the steps used in the algorithm:

- Run a backbone network (here, ResNet-101) over input image
- Add FCN to generate banks of $$k^2$$ position-sensitive score maps for each category, i.e. $$k^2(C+1)$$ output where $$k^2$$ represents the number of relative positions to divide an object (e.g. $$3^2$$ for a 3 by 3 grid) and C+1 representing the number of classes plus the background.
- Run a fully convolutional region proposal network (RPN) to generate regions of interest (RoI’s)
- For each RoI, divide it into the same $$k^2$$ “bins” or subregions as the score maps
- For each bin, check the score bank to see if that bin matches the corresponding position of some object. This process is repeated for each class.
- Once each of the $$k^2$$ bins has an “object match” value for each class, average the bins to get a single score per class
- Classify the RoI with a softmax over the remaining (C+1) dimensional vector

In short, Region Proposal Network (RPN), which is a fully convolutional architecture is used to extract candidate regions. Given the proposal regions (RoIs), the R-FCN architecture is designed to classify the RoIs into object categories and background.

-rfcn.png

### Position-sensitive score maps and Position-sensitive ROI pooling

The last convolutional layer of  produces a bank of $$k^2$$ position-sensitive score maps for each category, and thus has a $$k^2(C+1)$$ -channel output layer with C object categories (+1 for background). The bank of $$k^2$$ score maps correspond to a k x k spatial grid describing relative positions. For example, with k x k = 3 x 3, the 9 score maps encode the cases of {top-left, top-center, top-right, ..., bottom-right} of an object category.

-rfcn_maps.png

When ROI pooling, (C+1) feature maps with size of $$k^2$$ are produced, i.e. $$k^2(C+1)$$. The pooling is done in the sense that they are pooled with the same area and the same color in the figure. Average voting is performed to generate (C+1) 1d-vector. And finally softmax is performed on the vector.

Consider for example following example of R-FCN detecting a baby, 

-rfcn_roi.png

As [Joyce Xu](https://towardsdatascience.com/@joycex99) explains above example as,

> Simply put, R-FCN considers each region proposal, divides it up into sub-regions, and iterates over the sub-regions asking: “does this look like the top-left of a baby?”, “does this look like the top-center of a baby?” “does this look like the top-right of a baby?”, etc. It repeats this for all possible classes. If enough of the sub-regions say “yes, I match up with that part of a baby!”, the RoI gets classified as a baby after a softmax over all the classes.


### Advantages over Faster R-CNN

- The result is achieved at a test-time speed of 170ms per image. Faster than Faster R-CNN.
- Comparable detection quality (mAP) to Faster R-CNN

## SSD

[SSD](https://arxiv.org/pdf/1512.02325.pdf) is simple relative to previous methods that require object proposals because it completely eliminates proposal generation (*wooh*) and subsequent pixel or feature resampling stages and encapsulates all computation in a single network(*yay*). Hence, the name single shot detector (SSD). One model to solve them all. Simply remarkable. Let's analyse the steps used in the algorithm:

- Pass the image through a series of convolutional layers, yielding several sets of feature maps at different scales (e.g. 10x10, then 6x6, then 3x3, etc.)
- For each location in each of these feature maps, use a 3x3 convolutional filter to evaluate a small set of default bounding boxes. These default bounding boxes are essentially equivalent to Faster R-CNN’s anchor boxes.
- For each box, simultaneously predict a) the bounding box offset and b) the class probabilities
- During training, match the ground truth box with these predicted boxes based on IoU. The best predicted box will be labeled a “positive,” along with all other boxes that have an IoU with the truth >0.5.

-ssd.png

To put simply, SSD approach is based on a feed-forward convolutional network that produces a fixed-size collection of bounding boxes and scores for the presence of object class instances in those boxes, followed by a non-maximum suppression step to produce the final detections.

### Choosing scales and aspect ratios for default boxes

There are “extra feature layers” as seen in above architecture at the end that scale down in size. These varying-size feature maps help capture objects of different sizes, where each feature map is associated with a set of default bouding boxes. At each feature map cell, network predict the offsets relative to the default box shapes in the cell, as well as the per-class scores that indicate the presence of a class instance in each of those boxes. Specifically, for each box out of k at a given location, network computes c class scores and the 4 offsets relative to the original default box shape. This results in a total of (c + 4)k filters that are applied around each location in the feature map, yielding (c + 4)kmn outputs for a m × n feature map. Default boxes are similar to the anchor boxes used in Faster R-CNN only they are applied them to several feature maps of different resolutions.

-ssd_feature_map.png

Consider above example where, SSD evaluates a small set (e.g. 4) of default boxes of different aspect ratios at each location in several feature maps with different scales (e.g. 8 x 8 and 4 x 4 in middle and right images). For each default box, SSD predict
both the shape offsets and the confidences for all object categories belonging to C categories. At training time, SSD first  match these default boxes (middle and right) to the ground truth boxes (left image). For example, SSD have matched two default boxes with the cat and one with the dog, which are treated as positives and the rest as negatives.


### Challenges in Training

- Hard Negative Mining

After matching, wherein authors match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5), most of the default boxes are negatives, especially when the number of possible default boxes is large. This introduces a significant imbalance between the positive and negative training examples. Instead of using all the negative examples as seen from above example which can be a lot in proportion to positive, authors sort them using the highest confidence loss for each default box and pick the top ones so that the ratio between the negatives and positives is at most 3:1.

- Data Augmentation

Data augmentation is crucial. To make the model more robust to various input object sizes and shapes, each training image is randomly sampled by one of the following options:  use the entire original input image or sample a patch so that the minimum jaccard overlap with the objects is 0.1, 0.3, 0.5, 0.7, or 0.9. or randomly sample a patch. The size of each sampled patch is [0.1, 1] of the original image size, and the aspect ratio is between $$\frac{1}{2}$$ and 2. An improvement of 8.8% mAP is observed due to this strategy.

The model loss is a weighted sum between localization loss (e.g. Smooth L1) and confidence loss (e.g. Softmax).

### Advantages over Faster R-CNN

- The real-time detection speed is just astounding and way way faster (59 FPS with mAP 74.3% on VOC2007 test, vs. Faster R-CNN 7 FPS)
- Better detection quality (mAP) than any before
- Single network to solve them all (*Finally*)


## YOLO

- v1

- v2

- v3



## RetinaNet




## Backbones


- **MobileNet**





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

[R-FCN](https://arxiv.org/pdf/1605.06409.pdf)

[SSD](https://arxiv.org/pdf/1512.02325.pdf)

YOLO [v1]() [v2]() [v3]()

[RetinaNet]()

[Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)


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

