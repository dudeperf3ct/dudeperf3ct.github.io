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
  - [Vectorization](#vectorization)
    - [Bag of Words Model](#bag-of-words-model)
    - [Count Vectorizer](#count-vectorizer)
    - [TF-IDF Vectorizer](#tf-idf-vectorizer)
    - [N-grams Model](#n-grams-model)
  - [Embeddings](#embeddings)
    - [Word2Vec](#word2vec)
    - [Skip Gram Model](#skip-gram-model)
    - [CBOW](#cbow)
    - [Glove](#glove)
    - [Fasttext](#fasttext)
  - [LSTM](#lstm)
  - [GRU](#gru)
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

We will understand more clearly by example where each sentence is considered a document,

Sentence 1: I came I saw

Sentence 2: I conquered

From these three sentences, our vocabulary is as follows:

{ I, came, saw, conquered }

To get our bags of words using count vectorizer, we count the number of times each word occurs in each sentence. In Sentence 1, "the" appears once, and "came" and "saw" each appear once, so the feature vector for Sentence 1 is:

Sentence 1: { 2, 1, 1, 0 }

Similarly, the features for Sentence 2 are: { 1, 0, 0, 1 }


#### TF-IDF Vectorizer

Count Vectorizer tend to give higher score to more dominant words from the document but they may not contain "informational content" as much as rarer but domain specific words. For example, "I" from above example. Hence, we introduce TF-IDF. TF-IDF stands for term frequency-inverse document frequency. It gives a score as to how important a word is to the document in a corpus. TF-IDF measures relevance, not frequency. Wordcounts are replaced with TF-IDF scores across the whole corpus.The scores have the effect of highlighting words that are distinct (contain useful information) in a given document. The IDF of a rare term is high, whereas the IDF of a frequent term is likely to be low.

- Term Frequency: is a scoring of the frequency of the word in the current document.
- Inverse Document Frequency: is a scoring of how rare the word is across documents.

TF(t) = Number of times t appears in document / Total number of terms in document

IDF(t) = log(Total number of documents / Number of documents with term t in it)

TF-IDF(t) = TF(t) * IDF(t)

Let's look at our example,

Document 1 consists of Sentence 1: 

Term Frequency of Document 1 = {I: 2, came:1, saw: 1}

Term Frequency of Document 2 = {I: 1, conquered: 1}

Let's calculate TF-IDF("I") ,

TF("I") for Document 1 = 1/4

TF("I") for Document 2 = 1/2

IDF("I") for Document 1 = log(2/3)

TF-IDF("I") for Document 1 = (1/4) * log(2/3) 

TF-IDF("I") for Document 2 = (1/2) * log(2/3)

We get different weightings for same word.

#### N-gram Models

N-gram is contiguous sequence of n-items. Remember how using BoW we count occurrence of single word. Now, what if instead of using single word we used 2 consicutive words as construct bag-of-models from this model. We add the count based on the vocab used to construct a feature vector.

1-gram model (unigram), the new vocab will be { I, came, saw, conquered} same as BoW model vocabulary.

2-gram model (bigram), the new vocab will be { I came, came I, I saw, I conquered}

3-gram model (trigram), the new vocab will be {I came I, came I saw}

BoW can be considered as special case of n-gram model with n=1. 
Adding features of higher n-grams can be helpful in identifying that a certain multi-word expression occurs in the text.


#### Limitations of Bag-of-Words Model

- Vocabulary

The size of vocabulary requires careful design, most specifically in order to manage the size. The misspelling like come, cmoe will be considered as seperate words which can lead to increase in vocabulary.

- Sparsity

Sparse representations are harder to model both for computational reasons (space and time complexity). There will be a lot of zeros as input vectors will be one hot encoded of vocabulary size.

- Ordering

Discarding word order ignores the context, and in turn meaning of words in the document (semantics). Context and meaning can offer a lot to the model, that if modeled could tell the difference between the same words differently arranged (“this is interesting” vs “is this interesting”), synonyms (“old bike” vs “used bike”), and much more. This is one of the crucial drawbacks of using BoW models.


## Embeddings

Embeddings are the answer to mitigate the drawbacks of above model. Embeddings take into account the context and semantic meanings of words by producing a n-dimensional vector corresponding to that word. We will look into some of the popular ways of creating embeddings using different methods.

### Word2Vec

Ahh, the title. Word2Vec, converts a word to vector. But how? Word2vec is similar to an autoencoder, encoding each word in a vector. Word2Vec trains words against other words that neighbor them in the input corpus.


If the network is given enough training data (tens of billions of words), it produces word vectors with intriguing characteristics. Words with similar meanings appear in clusters, and clusters are spaced such that some word relationships, such as analogies, can be reproduced using vector math. The famous example is that, with highly trained word vectors, "king - man + woman = queen."

It comes in two flavors, the Continuous Bag-of-Words model (CBOW) and the Skip-Gram model. Algorithmically, these models are similar, except that CBOW predicts target words (e.g. 'mat') from source context words ('the cat sits on the'), while the skip-gram does the inverse and predicts source context-words from the target words.

#### Skip-Gram


However, skip-gram treats each context-target pair as a new observation, and this tends to do better when we have larger datasets. 



#### CBOW

CBOW smoothes over a lot of the distributional information (by treating an entire context as one observation). For the most part, this turns out to be a useful thing for smaller datasets. 


### Glove



### Fasttext






*We will look into CoVe, ELMo, ULMFit, GPT, BERT and GPT-2 models in the post on Transfer Learning in NLP.*


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

Must Read! [The Unreasonable Effectiveness of Recurrent Neural Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[Chater 9 Book: Speech and Language Processing by Jurafsky & Martin](https://web.stanford.edu/~jurafsky/slp3/9.pdf)

[Stanford CS231n Winter 2016 Chapter 10](https://www.youtube.com/watch?v=yCC09vCHzF8&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=10)

[CS224d slides and lectures](http://cs224d.stanford.edu/syllabus.html)

[Sebastian Raschka article on Naive Bayes](https://sebastianraschka.com/Articles/2014_naive_bayes_1.html)

[Word2Vec](https://arxiv.org/pdf/1301.3781.pdf)


---

# Footnotes and Credits


[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)



---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

