---
layout:     post
title:      Power of GAN
date:       2019-04-13 12:00:00
summary:    This post will provide a brief introduction of GANs.
categories: gan
published : false
---


# Generative Adversarial Networks

In this notebook, 

> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/RNN/char_rnn_keras.ipynb), [PyTorch](https://github.com/dudeperf3ct/DL_notebooks/blob/master/RNN/char_rnn_pytorch.ipynb) and [Fastai](https://github.com/dudeperf3ct/DL_notebooks/blob/master/RNN/char_rnn_fastai.ipynb).

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.

<p align="center">
<img src='/images/rnn/rnn_meme.jpg' width="50%"/> 
</p>


Feel free to jump anywhere,

- [Introduction to GAN](#introduction-to-gan)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)



# Introduction to GAN

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' /> 
</p>


<span class='red'>I-know-everything:</span> In our last post, we saw how we can use Adversarial Machine Learning in context of security. We discussed how adversaries can abuse the model and produce malicious results in real world. This name "Adversarial" has different meaning depending on the context. The previous post used Adversarial Training where neural network is used to correctly classify adversarial examples by training the network on adversarial examples. In context of RL, "self play" can be seen as Adversarial Training where the network learns to play with itself. In our today's topic, which is GAN i.e. Generative Adversarial Networks, we will use Adversarial Training where a model is trained on the inputs produced by adversary. Now all the names are cleared, let's get back to the revolutionary GANs. As all posts with GANs start with [quoting](https://www.quora.com/What-are-some-recent-and-potentially-upcoming-breakthroughs-in-deep-learning) [Yann LeCunn](http://twitter.com/ylecun/), "The most important one, in my opinion, is adversarial training (also called GAN for Generative Adversarial Networks). This is an idea that was originally proposed by Ian Goodfellow when he was a student with Yoshua Bengio at the University of Montreal (he since moved to Google Brain and recently to OpenAI).This, and the variations that are now being proposed is the most interesting idea in the last 10 years in ML, in my opinion." Tradition not broken. 

<span class='green'>I-know-nothing:</span> Holding up the tradition, what are GANs? Specifically, what does generative mean?

<span class='red'>I-know-everything:</span> Consider the example where are we are teaching a model to distinguish between dog(y=0) and cat(y=1). The way traditional machine learning classification algorithms like logistic regression or perceptron algorithm tries to find a straight line - a decision boundary - such that when a test image of dog is passed, model checks on which side of decision boundary does the animal falls and makes prediction accordinly. This is what we have learned so far from our past journey, nothing new. But consider a different approach. Instead of learning a decision boundary, what if network learns a model of dog looking at lots of different images of dog and same with cat, a different model of what a cat looks like. Now, when we pass a test image of dog to classify, the test image is matched against the model of dog and model of cat to make prediction. What's difference? Algorithms like logistic regression and perceptron learns mappings directly from space of inputs $$\chi$$ to labels {0, 1} i.e. p(y|x) where y $$\in$$ {0, 1} and these are called <span class='purple'>discriminative learning algorithms.</span> Now, instead of learning the mapping from input space, what if model learns the distribution of features of the labels? This is the idea behind <span class='purple'>generative learning algorithms.</span> They learn p(x|y), for example, if y=0 indicates it's a dog then p(x|y=0) models the distribution of dog's features and p(x|y=1) modelling the distribution of cat's features.




In next post, we will do something <span class='yellow'>different</span>. We will attempt to dissect any one or two papers. Any suggestions? So, let's call that Paper dissection.

<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology



---

# Further Reading

[NIPS 2016 Tutorial : Generative Adversarial Network](https://arxiv.org/pdf/1701.00160.pdf)

[Generative Learning algorithms](http://cs229.stanford.edu/notes/cs229-notes2.pdf)

[Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)

[WGAN]()

[DCGAN]()

[CycleGAN]()

[StyleGAN]()

[BigGAN]()

[Wasserstein GAN from depthfirstlearning](http://www.depthfirstlearning.com/2019/WassersteinGAN)

[Open Questions about Generative Adversarial Networks](https://distill.pub/2019/gan-open-problems/)

[Image Completion with Deep Learning in TensorFlow](http://bamos.github.io/2016/08/09/deep-completion/#step-1-interpreting-images-as-samples-from-a-probability-distribution)

[From GAN to WGAN](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)

[A Beginner's Guide to Generative Adversarial Networks (GANs)](https://skymind.ai/wiki/generative-adversarial-network-gan#zero)

[Understanding Generative Adversarial Networks](https://danieltakeshi.github.io/2017/03/05/understanding-generative-adversarial-networks/#fn:goodfellow)

[Understanding Generative Adversarial Networks (GANs)](https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29)

---

# Footnotes and Credits


[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)


---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

