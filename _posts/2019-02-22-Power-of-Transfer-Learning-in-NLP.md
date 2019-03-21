---
layout:     post
title:      Power of Transfer Learning in NLP
date:       2019-02-22 12:00:00
summary:    This post will provide a brief introduction 
categories: nlp transfer learning
published : false
---


# Transfer Learning in NLP

In this notebook, .

> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/lstm_and_gru/lstm_and_gru_keras.ipynb), [PyTorch](https://github.com/dudeperf3ct/DL_notebooks/blob/master/lstm_and_gru/lstm_and_gru_pytorch.ipynb), [Flair](https://github.com/dudeperf3ct/DL_notebooks/blob/master/lstm_and_gru/lstm_and_gru_flair.ipynb) and [fastai](https://github.com/dudeperf3ct/DL_notebooks/blob/master/lstm_and_gru/lstm_and_gru_fastai.ipynb).

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.


Feel free to jump anywhere,

- [](#nlp-tasks)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)

# NLP Tasks

### POS


### NER


### QA


### Coreference 



# Transfer Learning in NLP

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' /> 
</p>


<span class='red'>I-know-everything:</span> 


<span class='green'>I-know-nothing:</span> 


<span class='red'>I-know-everything:</span> 

The [embedding models](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#embeddings) which we disscused earlier like [word2vec](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#word2vec), [GLoVe](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#glove) and [fastText](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#fasttext) are fantastic in capturing meaning of individual words and their relationships by leveraging large datasets. These model generate word vectors of n-dimension which is used by neural network as starting point of training. The word vectors can be initialized to lists of random numbers before a model is trained for a specific task, or initialized with word vectors obtained from above embedding models.





## CoVe

In NLP tasks, context matters. That is, understanding context is very essential to all NLP tasks as words rarely appear in isolation. One such example is in Question Answering where understanding of how words in question shift the importance of words in document or in Summarization where model needs to understand which words capture the context clearly to summarize succinctly. The ability to share a common representation of words in the context of sentences that include them could further improve transfer learning in NLP. This is where CoVe comes into play which transfers information from large amounts of unlabeled training data in the form of word vectors has shown to improve performance over random word vector initialization on a variety of downstream tasks e.g. POS, NER and QA. 






## ELMO



## ULMFiT




## GPT



## BERT




## GPT-2





<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology


---

# Further Reading



---

# Footnotes and Credits


[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)



---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

