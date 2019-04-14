---
layout:     post
title:      Mystery of Adversarial Learning
date:       2019-03-04 12:00:00
summary:    This post will provide a brief introduction to 
categories: adversarial learning
published : false
---


# Mystery of Adversarial Learning

In this notebook, 

> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Adversarial%20Learning/adv_learning_keras.ipynb), [PyTorch](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Adversarial%20Learning/adv_learning_pytorch.ipynb) and [Tensorflow](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Adversarial%20Learning/adv_learning_tf.ipynb).

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.

<p align="center">
<img src='/images/rnn/rnn_meme.jpg' width="50%"/> 
</p>


Feel free to jump anywhere,

- [Adversarial Learning](#adversarial-learning)
  - [Clever Hans](#clever-hans)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)


# Adverserial Training

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' /> 
</p>


<span class='red'>I-know-everything:</span> My dear young padwan, you are learning quite a lot and honestly, keeping up with the pace of reasearch is really a challenge in itself. But we have tried to cover and focus on the basic concepts in Deep Learning and its application. The journey so far has been like visting a lot of cool stations. We visited MLP, CNN, RNN, LSTM & GRU, and also we visited one the very famous places Transfer Learning in CNN and Transfer Learning in NLP. Hope you enjoyed the journey. A lot is still waiting to be explored. One such topic of interest today is about Adveserial Training. But before that let me tell you a story about clever horse named Clever Hans who could do arithmetic.

<span class='green'>I-know-nothing:</span> Horse doing arithmetic? For real?  

<span class='red'>I-know-everything:</span> Here is the picture of clever hans (a horse) with its owner. The story of Clever Hans goes like this: 

<p align="center">
<img src='/images/adv_learning/cleverhans.jpg' width="50%"/> 
</p>

### Clever Hans

There was a horse named Clever Hans who could do arithmetic (*Yes you read it right*). The trainer of horse was not any kind of charlatan, didn't want to make any money of any sort but gained a lot of attention but also believed it to be true. You could ask the horse to do *2+2* and then the horse would tap his foot 4 times. Huge crowd of people would gather and watch the horse perform. As far as anybody could tell, it really was able to actually answer wide variety of questions people ask the horse. Later a pyschologist decided to examine the horse. He went to an enclosed area with no other people and wore a mask and he asked the horse to do arithmetic and the horse couldn't do it at all. It turned out what was happening that the horse had not learned arithmetic but *horse had learned how to read people's emotional reactions*. So, you would ask the horse add 1 plus 2 and horse would tap his hoof once and crowd would stare at him in expectation. Then the horse would tap his hoof the second time and everybody in the crowd would sit on the edge and see, then he would tap his hoof the third time and everybody would be like oh my good he knows arithmetic. And then he would stop taping.

So, clever hans was trained to answer these questions. And he found a way of doing it that made him appear to be successful by metric of "Can he provide the right answer where roomful of people are watching him?". He hadn't learned arithmetic and so could not generalize to unusual situations when there weren't a room of people to provide the reactions that he needed to solve the problem. 

<span class='green'>I-know-nothing:</span> Ahh now I see, the horse was clever indeed in reading people's emotional reactions. But how does this relate to Machine Learning I wonder?

<span class='red'>I-know-everything:</span> Great! So, we have seen how a vision model, where we use CNN, was able to attain human level accuracy in classifying images and we also saw what the model is actually looking at when it is classifying the image. If we given the following image of "Pig" to the classifer it correctly predicts that image as "Pig" but if we add a little(*calculated and not random*) noise to the same image, as you can see the resultant image(*original image + noise*) isn't much different from the original Pig image. If we pass this resultant image to the classifier, it predicts the image as "Airline" with 99% confidence.(😞) This resultant image is called "Adversarial Example". This example is *fooling CNN into thinking that Pig is a Airline.*

<p align="center">
<img src='/images/adv_learning/pig.png' width="50%"/> 
</p>

An Adversarial Example is an example that has been carefully computed to be misclassified. To make a new image indistinguishable to human obeserver from original image.
 
<span class='green'>I-know-nothing:</span> So what is really going on? Did the classifier cheat with us the same way Clever hans did? Are there any other methods which we can cheat? Is there any way to defend this cheating? Is it only in images or also in other tasks such as NLP and RL? This cheating can really put the state of the art classifier in a very difficult position as to are they really state of the art(SOTA) in classification and if someone misuses these techniques in fooling the classifier. This certainly has some serious after effects.

<span class='red'>I-know-everything:</span> That is certainly true. This issue of adversarial example does put the mark of SOTA  on classifier really in a jeopardy.

There are mainly 3 types of adversarial attacks. We will explain why is it so easy to perform them, and discuss the security implications that stem from these attacks.

1. Adversarial Inputs
2. Data poisoning Attacks
3. Model stealing Techniques

## Adversarial Attacks

We study each attack in-detail here.

### Adversarial Inputs


### Data poisoning Attacks


### Model stealing Techniques






<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology


---

# Further Reading

Stanford CS231n 2017 [Lecture 16 | Adversarial Examples and Adversarial Training](https://www.youtube.com/watch?v=CIfsB_EYsVI)


---

# Footnotes and Credits


[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)



---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

