---
layout:     post
title:      Force of Recurrent Neural Networks
date:       2019-01-28 12:00:00
summary:    This post will provide an brief introduction to recurrent neural networks and look at the results obtained by training Character RNN on various datasets.
categories: rnn
published : false
---


# Long-Short Term Memory and Gated Recurrent Units

In this notebook, we will see if Neural Networks can write as good as Shakespeare?

> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/RNN/char_rnn_keras.ipynb) and [PyTorch](https://github.com/dudeperf3ct/DL_notebooks/blob/master/RNN/char_rnn_pytorch.ipynb).

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.

<p align="center">
<img src='/images/rnn/rnn_meme.jpg' width="50%"/> 
</p>


Feel free to jump anywhere,

- [Introduction to LSTM and GRU](#introduction-to-lstm-and-gru)
  - [Backpropogation through time](#bptt)
  - [Character-Level Language Models](#character-level-language-models)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)


# Introduction to LSTM and GRU

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' /> 
</p>


<span class='red'>I-know-everything:</span>  So, Padwan today we are going to study about language through the lens of Neural Networks. Let me get philosophical for a bit, and show how we are what today because we are able to communicate which each other the ideas, the ideas to push the human race forward. Language has been a critical cornerstone to the foundation of human mankind and will also play a critical role in human-computer world. (*Blocking terminator vibes, transmitting [JARVIS](https://marvel-movies.fandom.com/wiki/J.A.R.V.I.S.) vibes....*)  

<span class='green'>I-know-nothing:</span> Does this mean that it will be the case where image where computer understand only numbers, the underlying language will also be converted to numbers and where some neural network does it's magic?



In next post, we explore the shortcomings of RNN by introducing <span class='purple'>Force of LSTM and GRU</span>.

<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology

Force of RNN - Recurrent Neural Networks

loss function - cost, error or objective function

jar jar backpropogation - backpropogation

jar jar bptt - BPTT

BPTT - backpropogation through time

---

# Further Reading

Must Read! [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

[Chater 9 Book: Speech and Language Processing by Jurafsky & Martin](https://web.stanford.edu/~jurafsky/slp3/9.pdf)

[Stanford CS231n Winter 2016 Chapter 10](https://www.youtube.com/watch?v=yCC09vCHzF8&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=10)

[CS224d slides and lectures](http://cs224d.stanford.edu/syllabus.html)



---

# Footnotes and Credits


[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)



---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

