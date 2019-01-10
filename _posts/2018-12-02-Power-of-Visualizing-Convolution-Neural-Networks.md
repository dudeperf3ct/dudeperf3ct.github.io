---
layout:     post
title:      Power of Visualizing Convolution Neural Networks
date:       2018-12-02 12:00:00
summary:    This post will provide an brief introduction to visualize trained CNN through transfer learning using Dogs vs Cats Redux Competition dataset from Kaggle along with the implementation in Keras framework.
categories: visualize cnn catsvsdogs
published : false
---


# Transfer Learning

In this notebook, we will try to answer the question "What CNN sees?" using [Cats vs Dogs Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) Competition dataset from kaggle. We will implement this using one of the popular deep learning framework <span class='yellow'>Keras</span> . 

> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Transfer%20Learning/transfer_learning_keras.ipynb), [PyTorch](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Transfer%20Learning/transfer_learning_pytorch.ipynb), and [fastai](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Transfer%20Learning/transfer_learning_fastai.ipynb).  

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but what is Transfer Learning?

Well sit tight and buckle up. I will go through everything in-detail.

--insert meme keanu reeves

Feel free to jump anywhere,

- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)




# Learning curves

Lets dive into interpreting learning different curves to understand and ways to avoid underfitting-overfitting or bias-variance tradeoff. There is some sort of tug of war between bias and variance, if we reduce bias error that leads to increase in variance error and vice-versa.

Let's recap what we had from our previous discussion on bias and variance.

- (Training - Dev) error high ==> High variance ==> Overfitting ==> Add more data to training set
- Training error high     ==> High bias    ===> Underfitting ==> Make model more complex
- Bayes error ==> Optimal Rate  ==> Unavoidable bias
- (Training - Bayes) error ===> Avoidable bias
- Bias = Optimal error rate (“unavoidable bias”) + Avoidable bias


## Cat Classifier

<p align="center">
<img src='/images/transfer_learning_files/cats_meme.jpg' />
</p>

Cats again! Suppose we run the algorithm using different training set sizes. For example, if you have 1,000 examples, we train separate copies of the algorithm on 100, 200, 300, ..., 1000 examples. Following are the different learning curves, where desired performance(green) along with dev(red) error and train(blue) error are plotted against the number of training examples.

Consider this learning curve,

<p align="center">
<img src='/images/transfer_learning_files/high_bias_or_variance.png' width="60%"/>
</p>


**Is this plot indicating, high bias, high variance or both?**

The training error is very close to desired performance, indicating avoidable bias is very low. The training(blue) error curve is relatively low, and dev(red) error is much higher than training error. Thus, the bias is small, but variance is large. As from recap above, adding more training data will help close gap between training and dev error and help reduce high variance.

Consider this curve,

<p align="center">
<img src='/images/transfer_learning_files/significant_bias_and_variance.png' width="60%"/>
</p>


**Is this plot indicating, high bias, high variance or both?**

This time, training error is large, as it is much higher than desired performance. There is significant avoidable bias. The dev error is also much larger than training error. This indicated we have significant bias and significant variance in our plot. We will use the ways to avoid both variance and bias.


Consider this curve,

<p align="center">
<img src='/images/transfer_learning_files/high_avoidable_bias.png' width="60%"/>
</p>

**Is this plot indicating, high bias, high variance or both?**

The training error is much higher than desired performance. This indicates it has high avoidable bias. The gap between training and dev error curves is small, indicating small variance.


**Lessons:**

- As we add more training data, training error can only get worse. Thus, the blue training error curve can only stay the same or go higher, and thus it can only get further away from the (green line) level of desired performance.

- The red dev error curve is usually higher than the blue training error. Thus, there’s almost no way that adding more data would allow the red dev error curve to drop down to the desired level of performance when even the training error is higher than the desired level of performance.


#### Techniques to reduce avoidable bias

- <span class='saddlebrown'>Increase model size (number of neurons/layers)</span>

This technique reduces bias by fitting training set better. If variance increases, we can use regularization to minimize the effect of increase in variance.

- <span class='saddlebrown'>Modify input features based on insights from error analysis</span>

Create additional features that help the algorithm eliminate a particular category of errors.These new features could help with both bias and variance.

- <span class='saddlebrown'>Reduce or eliminate regularization (L2 regularization, L1 regularization, dropout)</span>

This will reduce avoidable bias, but increase variance.

- <span class='saddlebrown'>Modify model architecture (such as neural network architecture)</span>

This technique can affect both bias and variance.

#### Techniques to reduce variance

- <span class='saddlebrown'>Add more training data</span>

This is the simplest and most reliable way to address variance, so long as you have access to significantly more data and enough computational power to process the data.

- <span class='saddlebrown'>Add regularization (L2 regularization, L1 regularization, dropout)</span>

This technique reduces variance but increases bias.

- <span class='saddlebrown'>Add early stopping (i.e., stop gradient descent early based on dev set error)</span>

This technique reduces variance but increases bias.

- <span class='saddlebrown'>Feature selection to decrease number/type of input features</span>

This technique might help with variance problems, but it might also increase bias. Reducing the number of features slightly (say going from 1,000 features to 900) is unlikely to have a huge effect on bias. Reducing it significantly (say going from 1,000 features to 100—a 10x reduction) is more likely to have a significant effect, so long as you are not excluding too many useful features. In modern deep learning, when data is plentiful, there has been a shift away from feature selection, and we are now more likely to give all the features we have to the algorithm and let the algorithm sort out which ones to use based on the data. But when your training set is small, feature selection can be very useful.

- <span class='saddlebrown'>Modify model architecture and modify input features</span>

These techniques are also mentioned in avoidable bias.

### Data Mismatch

Suppose a ML in a setting where the training and the dev/test distributions are different. Say, the training set contains Internet images + Mobile images, and the dev/test sets contain only Mobile images.

If the model generalizes well to new data drawn from the same distribution as the training set, but not to data drawn from the dev/test set distribution. We call this problem <span class='orange'>data mismatch</span> , since it is because the training set data is a poor match for the dev/test set data.

This illustration explains clearly the data mismatch problem.

<p align="center">
<img src='/images/transfer_learning_files/data_mismatch.png' />
</p>

- **Training set**

This is the data that the algorithm will learn from (e.g., Internet images + Mobile images). This does not have to be drawn from the same distribution as what we really care about (the dev/test set distribution).

- **Training dev set**

This data is drawn from the same distribution as the training set (e.g.,Internet images + Mobile images). This is usually smaller than the training set; it only needs to be large enough to evaluate and track the progress of our learning algorithm.

- **Dev set** 

This is drawn from the same distribution as the test set, and it reflects the distribution of data that we ultimately care about doing well on. (E.g., mobile images.)

- **Test set**

This is drawn from the same distribution as the dev set. (E.g., mobile images.)

### Techniques to resolve data mismatch problem

1. <span class='saddlebrown'>Try to understand what properties of the data differ between the training and the dev set distributions.</span>

2. <span class='saddlebrown'>Try to find more training data that better matches the dev set examples that your algorithm has trouble with.</span>

*In next post, we will discuss about various regularization techniques and when and how to use them. Stay tuned!*

# Introduction to Visualizing CNN

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' />
</p>


<span class='red'>I-know-everything:</span> Yo, my fellow apperentice. This time you will experience the <span class="purple"> Power of Transfer Learning</span>. Transfer Learning is a technique where you take a pretrained model trained on large dataset and transfer the learned knowledge to another model with small dataset but some what similar to large dataset for task of classification. For e.g. if we consider Imagenet dataset which contains 1.2 million images and 1000 categories, in that there are 24 different categories of dogs and 16 different categories of cats. So, we can transfer the learned features of cats and dogs from model trained on Imagenet dataset to our new model which contains 25,000 images of dogs and cats in training set.


<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology


Power of Transfer Learning - Transfer Learning

Power of Visualize CNN - Visualize CNN

ConvNets - Convolution Neural Networks

neurons - unit

Googlion planet - Google

---

# Further Reading

[Visualizaing and Understanding Convolution Neural Networks](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)


---

# Footnotes and Credits

[Kaggle Dataset for Cats and Dogs](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)


