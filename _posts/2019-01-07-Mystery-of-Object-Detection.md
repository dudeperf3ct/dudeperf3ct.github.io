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

For example consider we get an output of [0.1, 0.5, 0.4] (cat, dog, mouse) where the actual or expected output is [1, 0, 0] i.e. it is a cat. But our model predicted that given input has only 10% probability of being a cat, 50% probability of being dog and 40% of chance being a mouse. This being a multi-class classification, we can calculate the cross entropy using the formula for $$\mathbf{L_{mce}}$$ below. 

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

There are also other loss functions like Focal Loss(which we define in our RetinaNet), SVM Loss(Hinge), KL Divergence, Huber Loss etc.

*In next post, we will discuss some popular loss functions and where are they used. Stay tuned!*

# Introduction to Object Detection

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' />
</p>


<span class='red'>I-know-everything:</span> 

<span class='green'>I-know-nothing:</span> 

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

---

# Footnotes and Credits


[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)



---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

