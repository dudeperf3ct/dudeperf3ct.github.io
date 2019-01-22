---
layout:     post
title:      Magic of Style Transfer
date:       2018-12-23 12:00:00
summary:    This post will provide an brief introduction to different neural style transfer methods with some examples.
categories: style transfer
published : false
---


# Neural Style Transfer

In this notebook, we will try to answer the question "What CNN sees?" using [Cats vs Dogs Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) Competition dataset from kaggle. We will implement this using one of the popular deep learning framework <span class='yellow'>Keras</span> . 

> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Vis%20CNN/vis_cnn_keras.ipynb) and [fastai](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Vis%20CNN/vis_cnn_fastai.ipynb).  

> *All codes can be run on Google Colab (link provided in notebook).* [Awesome Tensorflow Library](https://github.com/PAIR-code/saliency/blob/master/Examples.ipynb) by PAIR Initiative, [Lucid](https://github.com/tensorflow/lucid) by Tensorflow, [Keras Vis Library](https://raghakot.github.io/keras-vis/) and [PyTorch](https://github.com/utkuozbulak/pytorch-cnn-visualizations). Try to use libraries instead of writing all the code from scratch. Not that it is bad practise. But we can see the same thing by standing on the the shoulder of giants (less work!). 

Hey yo, but how to see what a CNN sees?

Well sit tight and buckle up. I will go through everything in-detail.

<p align="center">
<img src='/images/visualize_cnn_files/keanu_meme.jpg' />
</p>

Feel free to jump anywhere,
- [Regularization](#regularization)
  - [Why does Regularization Work?](#why-does-regularization-work?)
  - [Tips for using Weight Regularization](#tips-for-using-weight-regularization)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)




# Regularization

Regualarization, is that another one of the fancy names, to look cooler? After introducing the bias and variance, overfitting and underfitting, ways of interpreting learning curves, now comes the time put all these pieces together. We learned to interpret if our model is overfitting or underfitting from learning curves in our last post on [Transfer Learning](https://dudeperf3ct.github.io/transfer/learning/catsvsdogs/2018/11/20/Power-of-Transfer-Learning/). So, now we will look into ways of how to handle these anomalies. First, why to use regularizations? All along in machine learning, we tried to make an algorithm that does good not only on training data, but also on new inputs i.e. to generalize data other than training or the unseen data.

If you suspect the model is overfitting (high variance), we call in regularization to rescue. We looked other ways we can do, like adding more data, which is not always the case as it can be expensive to get more data, and so on. So, adding regularization often helps in reducing overfitting (reduce variance). Good regularizers reduces variance significantly while not overly increasing bias.


<p align="center">
<img src='/images/visualize_cnn_files/overfitting.jpg' />
</p>

Michael Neilsen explains clearly the relation between parameters in model and generalizability,

> Models with a large number of free parameters can describe an amazingly wide range of phenomena. Even if such a model agrees well with the available data, that doesn't make it a good model. It may just mean there's enough freedom in the model that it can describe almost any data set of the given size, without capturing any genuine insights into the underlying phenomenon. When that happens the model will work well for the existing data, but will fail to generalize to new situations. The true test of a model is its ability to make predictions in situations it hasn't been exposed to before.

Here we will dive deep into two well-know regularizers ($$L^1$$ and $$L^2$$) and in next post discuss the remaining ones.

**$$L^2$$ regularization:** This regularization goes by many names, **Ridge regression**, **Tikhonov regularization**, **Weight Decay** or **Fobenius Norm** (used in different contexts). This method imposes a penalty by adding a regularization term $$\Omega(\theta) = \lambda \left\Vert\left(\mathbf{w}\rm\right) \right\Vert^2$$ to the objective(loss) function. Here, $$\lambda$$ is the regularization parameter. It is the hyperparameter whose value is optimized for better results. The penalty tends to drive all the weights to smaller values.

From Michael Neilsen on $$L^2$$ regularization, 
> Intuitively, the effect of regularization is to make it so the network prefers to learn small weights, all other things being equal. Large weights will only be allowed if they considerably improve the first part of the cost (loss or objective) function. Put another way, regularization can be viewed as a way of compromising between finding small weights and minimizing the original cost function. The relative importance of the two elements of the compromise depends on the value of $$\lambda$$: when $$\lambda$$ is small we prefer to minimize the original cost function, but when $$\lambda$$ is large we prefer small weights.

**$$L^1$$ regularization:** This regularization goes by one other name, **LASSO regression** (least absolute shrinkage
and selection operator). This method imposes a penalty by adding a regularization term $$\Omega(\theta) = \lambda ||\mathbf{w}||$$ to the objective(loss) function. Here, $$\lambda$$ is the regularization parameter. It is the hyperparameter whose value is optimized for better results. The penalty tends to drive some weights to exactly zero (introducing sparsity in the model), while allowing some weights to be big. <span class='red'>The key difference between these techniques is that LASSO shrinks the less important feature’s coefficient to zero thus, removing some feature altogether. So, this works well for feature selection in case we have a huge number of features.</span>

### Why does Regularization Work?

Now, the answer question which follows what is, is why? Why does it work? Consider an example below,

<p align="center">
<img src='/images/visualize_cnn_files/just_right.png'/>
</p>

Our goal is to build a model that lets us predict $$\mathbf{y}$$ as function of $$\mathbf{x}$$. First we will fit a polynomial model and then look into case of fitting neural networks. As there are 5 points in graph above, which means we can find a unique 4th-order polynomial $$\mathbf{y}=\mathbf{a_0}+\mathbf{a_1}\mathbf{x_1}+…+\mathbf{a_4}\mathbf{x_4}$$ which fits the data exactly as shown in the graph(rightmost). But we can also get a good fit using the quadratic model $$\mathbf{y}=\mathbf{a_0}+\mathbf{a_1}\mathbf{x_1}+\mathbf{a_2}\mathbf{x_2}$$, as shown in graph(middle).

Now question is **Which of these is the better model? Which is more likely to be true? And which model is more likely to generalize well to other examples of the same underlying real-world phenomenon?**

It's not a priori possible to say which of these two possibilities is correct. (Or, indeed, if some third possibility holds). Logically, either could be true. And it's not a trivial difference. It's true that on the data provided there's only a small difference between the two models. But suppose we want to predict the value of y corresponding to some large value of x, much larger than any shown on the graph above. If we try to do that there will be a dramatic difference between the predictions of the two models.

> One point of view is to say that in science we should go with the simpler explanation, unless compelled not to. When we find a simple model that seems to explain many data points we are tempted to shout "Eureka!" After all, it seems unlikely that a simple explanation should occur merely by coincidence. Rather, we suspect that the model must be expressing some underlying truth about the phenomenon.

And so while the 4th order model works perfectly for these particular data points, the model will fail to generalize to other data points, and the noisy 2nd order model will have greater predictive power.

> Let's see what this point of view means for neural networks. Suppose our network mostly has small weights, as will tend to happen in a regularized network. The smallness of the weights means that the behaviour of the network won't change too much if we change a few random inputs here and there. That makes it difficult for a regularized network to learn the effects of local noise in the data. Think of it as a way of making it so single pieces of evidence don't matter too much to the output of the network. Instead, a regularized network learns to respond to types of evidence which are seen often across the training set. By contrast, a network with large weights may change its behaviour quite a bit in response to small changes in the input. And so an unregularized network can use large weights to learn a complex model that carries a lot of information about the noise in the training data. <span class='purple'>In a nutshell, regularized networks are constrained to build relatively simple models based on patterns seen often in the training data, and are resistant to learning peculiarities of the noise in the training data. The hope is that this will force our networks to do real learning about the phenomenon at hand, and to generalize better from what they learn.</span>

Michael Neilsen explains in his book concludes that, <span class='red'>"Regularization may give us a computational magic wand that helps our networks generalize better, but it doesn't give us a principled understanding of how generalization works, nor of what the best approach is."</span>

Here is an example from the book to understand the above statement,

> A network with 100 hidden neurons has nearly 80,000 parameters. We have only 50,000 images in our MNSIT training data. It's like trying to fit an 80,000th degree polynomial to 50,000 data points. By all rights, our network should overfit terribly. And yet, as we saw earlier, such a network actually does a pretty good job generalizing. Why is that the case? It's not well understood. It has been conjectured  that "the dynamics of gradient descent learning in multilayer nets has a 'self-regularization' effect". This is exceptionally fortunate, but it's also somewhat disquieting that we don't understand why it's the case. <span class='purple'>In the meantime, we will adopt the pragmatic approach and use regularization whenever we can. Our neural networks will be the better for it.</span>


### Tips for using Weight Regularization

- Use with All Network Types

Weight regularization is generic technique. It can be used with all types neural networks we saw uptil now, MLP, CNN, LSTM and RNN (which we will address in later posts).

- Standardize Input Data

It is generally good practice to update input variables to have the same scale. When input variables have different scales, the scale of the weights of the network will, in turn, vary accordingly. This introduces a problem when using weight regularization because the absolute or squared values of the weights must be added for use in the penalty. This problem can be addressed by either normalizing or standardizing input variables.

- Use Larger Network

It is common for larger networks (more layers or more nodes) to more easily overfit the training data. When using weight regularization, it is possible to use larger networks with less risk of overfitting. A good configuration strategy may be to start with larger networks and use weight decay.

- Grid Search Parameters

It is common to use small values for the regularization hyperparameter that controls the contribution of each weight to the penalty. Perhaps start by testing values on a log scale, such as 0.1, 0.001, and 0.0001. Then use a grid search at the order of magnitude that shows the most promise.

- Use L1 + L2 Together

Rather than trying to choose between L1 and L2 penalties, use both. Modern and effective linear regression methods such as the Elastic Net use both L1 and L2 penalties at the same time and this can be a useful approach to try. This gives you both the nuance of L2 and the sparsity encouraged by L1.

- Use on a Trained Network

The use of weight regularization may allow more elaborate training schemes. For example, a model may be fit on training data first without any regularization, then updated later with the use of a weight penalty to reduce the size of the weights of the already well-performing model.

*In next post, we will discuss about other regularization techniques and when and how to use them. Stay tuned!*

# Introduction to Visualizing CNN

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' />
</p>


<span class='red'>I-know-everything:</span> Today will be the exiciting topic of peeking inside of the black box of CNNs and look at what they see. So, to recap, from our previous posts we saw what a CNN is, wherein we trained CNN and looked at different architectures. Next, we moved to transfer learning. There we learned what an amazing technique transfer learning is! Different ways of transfer learning and how and why transfer learning provides such a boost. In that topic, we introduced to this notion of CNN as black box, where we really can't tell as to what is that network is looking at while training or predicting and how such amazing CNN learn to classify 1000 categories of 1.2 million images better than humans. So, today we will look behind the scenes of working of CNNs and this will involve looking at lots of pictures. 



<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology


Power of Transfer Learning - Transfer Learning

Power of Visualize CNN - Visualize CNN

ConvNets - Convolution Neural Networks

neurons - unit

cost function - loss or objective function

---

# Further Reading



---

# Footnotes and Credits


[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)

---
**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

