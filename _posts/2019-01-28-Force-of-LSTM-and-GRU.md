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
    - [N-gram Model](#n-gram-model)
  - [Embeddings](#embeddings)
    - [Word2Vec](#word2vec)
    - [Skip-gram Model](#skip-gram-model)
    - [CBOW](#cbow)
    - [GloVe](#glove)
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


<p align="center">
<img src='/images/lstm_and_gru/bag_of_words.png' /> 
</p>


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

Ahh, the title, Word2Vec, converts a word to vector. [Mikilov et al](https://arxiv.org/pdf/1301.3781.pdf) developed the word2vec toolkit that allows to use pretrained embeddings. But how? Word2vec is similar to an autoencoder, encoding each word in a vector. Word2Vec trains words against other words that neighbor them in the input corpus. Word2Vec consists of 3-layer neural network (not very deep) i.e. input layer, hidden layer and output layer. Depending on which model (skip-gram or cbow), we feed in word and train to predict neighbouring words or feed neighbouring words and train to predict missing word. Once we obtain trained model, we remove last output layer and when we input a word from vocabulary, output given by hidden layer will be "embedding of the input word".

If the network is given enough training data (tens of billions of words), it produces word vectors with intriguing characteristics. Words with similar meanings appear in clusters, and clusters are spaced such that some word relationships, such as analogies, can be reproduced using vector math. The famous example is that, with highly trained word vectors, "king - man + woman = queen."  Patterns such as “Man is to Woman as Brother is to Sister” can be generated through algebraic operations on the vector representations of these words such that the vector representation of “Brother” - ”Man” + ”Woman” produces a result which is closest to the vector representation of “Sister” in the model. Such relationships can be generated for a range of semantic relations (such as Country–Capital) as well as syntactic relations (e.g. present tense–past tense). A similar example of the result of a vector calculation vec(“Madrid”) - vec(“Spain”) + vec(“France”) is closer to vec(“Paris”) than to any other word vector.

It comes in two flavors, the Continuous Bag-of-Words model (CBOW) and the Skip-Gram model. Algorithmically, these models are similar, except that CBOW predicts target words (e.g. 'mat') from source context words ('the cat sits on the'), while the skip-gram does the inverse and predicts source context-words from the target words. 

#### Skip-gram Model

Skip-gram model predicts context (surrounding) words given the current word. The training objective is to learn word vector representations that are good at predicting the nearby words. To understand what that means, lets consider example shown below. 

Here we consider the window size = 2, window size refers to the number of words to be looked on either side of focus or input word. The highlighted blue color word is input and it produces training samples depending on context words.

<p align="center">
<img src='/images/lstm_and_gru/skip_gram_1.png' /> 
</p>

<p align="center">
<img src='/images/lstm_and_gru/skip_gram_2.png' /> 
</p>

The pairs to right are training samples. The training of skip-gram will take one-hot vector input on vocabulary and outputs a probability after applying Hierarchical softmax that the particular word is output given the input word. Given enough input vectors, model learns that there is high probability that when "San" is given as input, "Franciso" or "Jose" is more likely than "York". Skip-gram treats each context-target pair as a new observation, and this tends to do better when we have larger datasets. 


<p align="center">
<img src='/images/lstm_and_gru/skip_gram.png' /> 
</p>


#### CBOW Model

Continuous bag of words (CBOW) model predicts the current word based on several surrounding words. The training objective is to learn word vector representations that are good at predicting missing word given the nearby words.


Here we also consider the window size = 2.

<p align="center">
<img src='/images/lstm_and_gru/cbow_1.png' /> 
</p>

<p align="center">
<img src='/images/lstm_and_gru/cbow_2.png' /> 
</p>

The pairs to right are training samples. It's like all the pairs from above Skip-gram training samples are inverted. That's exactly how it is. CBOW smoothes over a lot of the distributional information (by treating an entire context as one observation). For the most part, this turns out to be a useful thing for smaller datasets. CBOW is faster while skip-gram is slower but does a better job for infrequent words. 

<p align="center">
<img src='/images/lstm_and_gru/cbow.png' /> 
</p>

#### Training Tricks

Word2Vect uses some tricks in training. We will look at some of them.

- Negative Sampling


- Subsampling Frequent Words

We have seen how training samples are created. In very large corpora, the most frequent words can easily occur hundreds of millions of times (e.g.,“in”, “the”, and “a”). Such words usually provide less information value than the rare words and being the most common word it occurs pretty much everywhere. So, how to learn a good vector for the word "the"? This where the authors propose a subsampling method for frequent words. The idea to change the vector representation of frequent words gradually. To counter the imbalance between the rare and frequent words, authors used a simple subsampling approach: each word $$\mathbf{w_{i}}$$ in the training set is discarded with probability computed by the formula $$ P(\mathbf{w_i}) = 1 - \sqrt{\frac{t}{\mathbf{f(\mathbf{w}_i)}}}$$ where $$\mathbf{f(\mathbf{w}_i)}}$$ is s  the  frequency  of  word $$\mathbf{w_i}$$ and t is a  chosen threshold, typically around $$10^-5$$.
 

- Hierarchical Softmax

The traditional softmax is very computationally expensive especially for large vocabulary size, typicall for vocabulary size of V the order is O(V), but hierarchical softmax reduces the computation to O(log(V)).




One of the biggest challenges with Word2Vec is how to handle unknown or out-of-vocabulary (OOV) words and morphologically similar words. This can particularly be an issue in domains like medicine where synonyms and related words can be used depending on the preferred style of radiologist, and words may have been used infrequently in a large corpus. If the word2vec model has not encountered a particular word before, it will be forced to use a random vector, which is generally far from its ideal representation. 

### GloVe



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

Must Read! [Edwin Chen's blog on LSTM]()

Must Read! Word Embeddings by Sebastian Ruder [part-1](http://ruder.io/word-embeddings-1/), [part-2]()

[Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)

[Chater 9 Book: Speech and Language Processing by Jurafsky & Martin](https://web.stanford.edu/~jurafsky/slp3/9.pdf)

[Stanford CS231n Winter 2016 Chapter 10](https://www.youtube.com/watch?v=yCC09vCHzF8&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=10)

[CS224d slides and lectures](http://cs224d.stanford.edu/syllabus.html)

[A brief history of word embeddings](https://www.gavagai.se/blog/2015/09/30/a-brief-history-of-word-embeddings/)

[Sebastian Raschka article on Naive Bayes](https://sebastianraschka.com/Articles/2014_naive_bayes_1.html)

[Distributed Representations of Words and Phrasesand their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

[Word2Vec](https://arxiv.org/pdf/1301.3781.pdf)

[GloVe](https://nlp.stanford.edu/pubs/glove.pdf)

[word2vec Explained](https://arxiv.org/pdf/1402.3722v1.pdf)

[Quora: How does word2vec works?](https://www.quora.com/How-does-word2vec-work-Can-someone-walk-through-a-specific-example)

---

# Footnotes and Credits


[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)



---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

