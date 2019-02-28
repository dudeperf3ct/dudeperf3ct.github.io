---
layout:     post
title:      Force of Recurrent Neural Networks
date:       2019-01-28 12:00:00
summary:    This post will provide an brief introduction to recurrent neural networks and look at the results obtained by training Character RNN on various datasets.
categories: rnn
published : false
---


# Long-Short Term Memory and Gated Recurrent Units

In this notebook, 

> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/RNN/char_rnn_keras.ipynb), [PyTorch](https://github.com/dudeperf3ct/DL_notebooks/blob/master/RNN/char_rnn_pytorch.ipynb) and.

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.


Feel free to jump anywhere,

- [Introduction to LSTM and GRU](#introduction-to-lstm-and-gru)
  - [](#)
  - [](#)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)

# Preprocessing Text





# Introduction to LSTM and GRU

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' /> 
</p>


<span class='red'>I-know-everything:</span> Today we will be visiting a lot of concepts in field of NLP. I mean a lot. There will be a lot to taken in so don't get lost (*in space*).

<span class='green'>I-know-nothing:</span> 

<span class='red'>I-know-everything:</span> Let me start with the various vectorization and embeddings techniques and gradually we will move into LSTM and GRUs.


In the [last post]() on RNNs we saw how neural networks only understand numbers and all we have as input is string of words which make up sentences, which add upto paragraphs and eventually make a document. Collection of such documents is called corpus. The text is converted to tokens using tokens and into numbers using vectorization/embeddings/numericalizations.

So, to convert the text we often take help of various techniques. Let's visit them one by one.

## Vectorization

Vectorization refers to the process of converting strings to numbers. These numbers which are then fed to neural networks to do their thing. There are various ways we can convert these strings into numbers. This process is also called feature extraction or feature encoding. In this techniques we will often encounter with the word **Vocabulary**, vocab is nothing but collection of unique words or characters depending on how we want. 

We will make this concrete with example.

Example Sentence: The cat sat on the mat.

vocab_character : {T, h, e, c, a, t, s, o, n, m, .}

vocab_words : {The, cat, sat, on, the, mat, .}

If we had converted all the text to lower, new vocabulary would have been

vocab_character : {t, h, e, c, a, s, o, n, m, .}

vocab_words : {the, cat, sat, on, mat, .}

Notice, the repeated "the" is now gone. Hence, unique collection of words or characters. Note, we will assume that our sentences will be lower case even though they may appear upper case.

### Bag-of-Words Model

This is one of the most simple and naive way to vectorize. As the name suggests, we are creating a bag of models. The simplest way to create a vocabulary is to bag uniques words(characters). 

Sentence 1: I came I saw

Sentence 2: I conquered

From these three sentences, our vocabulary is as follows:

{ I, came, saw, conquered }


#### Count Vectorizer

BoW Model learns a vocabulary from each document and model each document by counting the occurence of word in the document. This is done on top of Bag-of-Models. Here each word count is considered as feature vector. CountVectorizer works on Terms Frequency, i.e. counting the occurrences of tokens.

We will understand more clearly by example,

Sentence 1: I came I saw

Sentence 2: I conquered

From these three sentences, our vocabulary is as follows:

{ I, came, saw, conquered }

To get our bags of words using count vectorizer, we count the number of times each word occurs in each sentence. In Sentence 1, "the" appears once, and "came" and "saw" each appear once, so the feature vector for Sentence 1 is:

Sentence 1: { 2, 1, 1, 0 }

Similarly, the features for Sentence 2 are: { 1, 0, 0, 1 }


#### TF-IDF Vectorizer




#### N-gram Models


## Embeddings



### Word2Vec



#### Skipgram



#### CBOW



### Glove



## LSTM


## GRU



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

[Sebastian Raschka article on Naive Bayes](https://sebastianraschka.com/Articles/2014_naive_bayes_1.html)



---

# Footnotes and Credits


[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)



---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

