---
layout:     post
title:      Force of Recurrent Neural Networks
date:       2019-01-07 12:00:00
summary:    This post will provide an brief introduction to recurrent neural networks.
categories: rnn
published : false
---


# Recurrent Neural Networks

In this notebook, 

> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Object%20Detection/Keras/object_detection_keras.ipynb), [PyTorch](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Object%20Detection/PyTorch/object_detection_pytorch.ipynb), [Tensorflow](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Object%20Detection/Tensorflow/object_detection_tensorflow.ipynb), [fastai](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Object%20Detection/Fastai/object_detection_fastai.ipynb) and [Demos](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Object%20Detection/Demos).  

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.

-meme


Feel free to jump anywhere,

- [Introduction to Recurrent Neural Networks](#introduction-to-recurrent-neural-networks)
- [Recap](#recap)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)

Here, let me borrow some of the points from [Prof. Yuval Noah Harrari's](https://www.ynharari.com/about/) book on [Sapiens](https://www.amazon.com/Sapiens-A-Brief-History-Humankind/dp/1846558239/),

> <span class='purple'>How did Homo sapiens came to dominate the planet? The secret was a very peculiar characteristic of <i>our unique Sapiens language</i>. Our language, alone of all the animals, enables us to talk about things that do not exist at all. You could never convince a monkey to give you a banana by promising him limitless bananas after death, in monkey heaven.</span>

> <span class='red'>By combining profound insights with a remarkably vivid language, Sapiens acquired cult status among diverse audiences, captivating teenagers as well as university professors, animal rights activists alongside government ministers.</span>


# Introduction to Recurrent Neural Networks

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' /> 
</p>


<span class='red'>I-know-everything:</span>  So, Padwan today we are going to study about language through the lens of Neural Networks. Let me get philosophical for a bit, and show how we are what today because we are able to communicate which each other the ideas, the ideas to push the human race forward. Language has been a critical cornerstone to the foundation of human mankind and will also play a critical role in human-computer world.*Don't send terminator vibes, transmitting [JARVIS](https://marvel-movies.fandom.com/wiki/J.A.R.V.I.S.) vibes....*  

<span class='green'>I-know-nothing:</span> Does this mean that it will be like Image where computer understand only numbers, the underlying language will be converted to numbers and where neural network does it's magic?

<span class='red'>I-know-everything:</span> Bingo, that's exactly right. As Image understanding has it's own challenges like occlusion, viewpoint variation, deformations, background clutter, etc., dealing with language comes with it's own challenges, starting from what language we are dealing with. This is what Natural Language Processing (NLP) field is about.  

Let me jump and tell you about the idea of what is <span class='purple'> Force of RNN </span>. RNN known as recurrent neural networks or Elman Network are useful for dealing with sequences.

<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology

Force of RNN - Recurrent Neural Networks

loss function - cost, error or objective function


---

# Further Reading

Must Read! [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

[Chater 9 Book: Speech and Language Processing by Jurafsky & Martin](https://web.stanford.edu/~jurafsky/slp3/9.pdf)



[Recurrent neural network based language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)

[A Primer on Neural Network Modelsfor Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf)

[Extensions of Recurrent neural network language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2011/mikolov_icassp2011_5528.pdf)

[Wildml Introduction to RNN](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)


---

# Footnotes and Credits


[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)


---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

