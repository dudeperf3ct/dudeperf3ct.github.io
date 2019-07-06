---
layout:     post
title:      Fun of dissecting paper
date:       2019-04-29 12:00:00
summary:    This post will provide a brief introduction to meta-learning.
categories: meta learning
published : false
---


# Fun of Dissecting Paper

In this post, we will take a different approach in learning a topic. We will be looking at various papers in the topic of Learning to learn aka Meta-Learning but here we will provide a [curriculum](http://www.depthfirstlearning.com/), starting with introduction to the meta-learning and then diving into specifics of different algorithms in meta-learning and finally implementing them.

> All the codes implemented in Jupyter notebook in PyTorch [meta_learning_baseline](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Meta%20Learning/meta_learning_baseline.ipynb), [meta_learning_baseline++](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Meta%20Learning/meta_learning_baseline++.ipynb) and [meta_learning_maml](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Meta%20Learning/meta_learning_maml.ipynb)

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.

<p align="center">
<img src='/images/meta_learning/meta_meme.jpg' width="40%"/> 
</p>


Feel free to jump anywhere,

- [How to Learn Learning to Learn?](#how-to-learn-learning-to-learn)
  - [Week 1](#week-1)
  - [Week 2](#week-2)
  - [Week 3](#week-3)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)


Ah, you must be wondering why is <span class='red'>I-know-everything</span> not present to teach the disciple <span class='green'>I-know-nothing</span> dissect the papers. 

# How to Learn Learning to Learn?

This is an experiemental post or a different approach to learning a new topic. We will divide the task into a 3-week long learning journey. In the first week, we will focus on getting familiar with the term meta-learning and various terminologies associated with it. We will jump into 3 papers which explore meta-learning through metrics-based algorithms. In the following week, we will dive into some other types of learning algorithms namely model-based and optimization-based meta-learning algorithms and learn in-detail about them. In the third week, we will implement some of the algorithms we looked at in the previous week.

## Week 1  (Getting Started)

### Video & Slides

Video [Introductory talk by Oriol Vinyals](https://www.facebook.com/nipsfoundation/videos/1552060484885185/) if in hurry jump yo last 30 minutes and [Slides](http://metalearning-symposium.ml/files/vinyals.pdf)

Video [On Learning How to Learn Learning Strategies](https://vimeo.com/250399374)

Video [Siamese Network](https://www.coursera.org/lecture/convolutional-neural-networks/siamese-network-bjhmj) and [One Shot Learning](https://www.coursera.org/lecture/convolutional-neural-networks/one-shot-learning-gjckG)


### Blog 

[🐣 From zero to research — An introduction to Meta-learning](https://medium.com/huggingface/from-zero-to-research-an-introduction-to-meta-learning-8e16e677f78a)

[Meta-Learning: Learning to Learn Fast](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)

### Paper 

[Siamese Neural Networks for One-shot Image Recognition by Koch et al](http://www.cs.toronto.edu/~rsalakhu/papers/oneshot1.pdf) aka Siamese Neural Networks

[Matching networks for one shot learning by Vinyals et al](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) aka Matching Networks

[Learning to compare: Relation network for few-shot learning by Sung et al](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Sung_Learning_to_Compare_CVPR_2018_paper.pdf) aka Relation Networks

### Questions

- How is meta-learning different from supervised learning?

- How is dataset for training and testing setup different from typical setting?

- What does names like meta-training, meta-testing, support, query mean?

- What does "Go beyond train from samples from a single distribution" mean in meta-learning?


## Week 2 (Diving into specifics)

### Video & Slides

Video MILA Talks [Few-Shot Learning with Meta-Learning: Progress Made and Challenges Ahead - Hugo Larochelle](https://www.youtube.com/watch?v=b8JlilRnhM4) and [Slides](http://metalearning.ml/2018/slides/meta_learning_2018_Larochelle.pdf)

Video ICML 2019 [Meta-Learning: From Few-Shot Learning to Rapid Reinforcement Learning](https://www.facebook.com/icml.imls/videos/meta-learning-from-few-shot-learning-to-rapid-reinforcement-learning/400619163874853/)

Video NeurIPS 2017 [Panel discussion](https://vimeo.com/250399623)

Slides [Model-Agnostic Meta-Learning Universality, Inductive Bias, and Weak Supervision](http://metalearning.ml/2017/slides/metalearn2017_finn.pdf)

### Blog

[Learning to Learn](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)

[Reptile: A Scalable Meta-Learning Algorithm](https://openai.com/blog/reptile/)


### Paper

[Optimization as a model for Few-Shot Learning by Ravi & Larochelle](https://openreview.net/pdf?id=rJY0-Kcll) aka Meta-Learner LSTM

[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks by Finn et al](https://arxiv.org/pdf/1703.03400) aka MAML

[Prototypical Networks for Few-shot Learning](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf) aka ProtoNet

[On First-Order Meta-Learning Algorithms](https://arxiv.org/pdf/1803.02999) aka Reptile

### Book 

[Chapter 2: Meta Learning](https://link.springer.com/content/pdf/10.1007%2F978-3-030-05318-5.pdf)


### Questions

<p align="center">
<img src='/images/meta_learning/meta_taxonomy.png' width="80%"/> 
</p>

- How is Meta-Learner LSTM different from Matching Networks?

- How is MAML different from Meta-Learner LSTM?


## Week 3 (Coding Challenge)

### Video & Slides

Slides [What’s Wrong with Meta-Learning(and how we might fix it)](http://metalearning.ml/2018/slides/meta_learning_2018_Levine.pdf)

Slides [Meta-Learning Frontiers:Universal, Uncertain, and Unsupervised](http://people.eecs.berkeley.edu/~cbfinn/_files/metalearning_frontiers_2018_small.pdf)

### Paper

[Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples](https://arxiv.org/pdf/1903.03096.pdf)

[A Closer Look at Few-shot Classification](https://openreview.net/pdf?id=HkxLXnAcFQ)

### Challenge

Challenge for this week will be to implement 3 algorithms from the paper [A Closer Look at Few-shot Classification](https://openreview.net/pdf?id=HkxLXnAcFQ). We will implement baseline, baseline++ and MAML algorithms using Omniglot dataset and try to replicate the results shown in the paper.

### Code

**MAML**

[https://github.com/cbfinn/maml](https://github.com/cbfinn/maml)

[https://github.com/dragen1860/MAML-Pytorch](https://github.com/dragen1860/MAML-Pytorch)

[https://github.com/katerakelly/pytorch-maml](https://github.com/katerakelly/pytorch-maml)

**Prototypical Nets**

[https://github.com/jakesnell/prototypical-networks](https://github.com/jakesnell/prototypical-networks)

**Reptile**

[https://github.com/openai/supervised-reptile](https://github.com/openai/supervised-reptile)

**Relational Nets**

[https://github.com/floodsung/LearningToCompare_FSL]( https://github.com/floodsung/LearningToCompare_FSL)

**4 Few-Shot Classification Algorithms**

[https://github.com/wyharveychen/CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot)

### Datasets

[Omniglot](https://github.com/brendenlake/omniglot/)

[Mini-Imagenet](https://github.com/y2l/mini-imagenet-tools)

[Meta-Dataset](https://github.com/google-research/meta-dataset)


In next post, we will look work on a project of <span class='purple'>building a text recognizer application</span>.

<span class='orange'>Happy Learning!</span>

---

# Further Reading

[NeurIPS 2017 Meta-learning symposium](http://metalearning-symposium.ml/)

[NeurIPS 2017 Workshop on Meta-Learning](https://nips.cc/Conferences/2017/Schedule?showEvent=8767)

[Online Meta-Learning](https://arxiv.org/pdf/1902.08438)

[SNAIL Paper](https://openreview.net/forum?id=B1DmUzWAW)

[Learning to Optimize with Reinforcement Learning](https://bair.berkeley.edu/blog/2017/09/12/learning-to-optimize-with-rl/)

[Learning to Optimize](https://arxiv.org/pdf/1606.01885)

[Meta-Learning a Dynamical Language Model](https://arxiv.org/pdf/1803.10631)

[Learning Unsupervised Learning Rules](https://arxiv.org/pdf/1804.00222)

[Learning to learn by gradient descent by gradient descent](https://arxiv.org/pdf/1606.04474)

---

# Footnotes and Credits

[Meta meme](https://twitter.com/joavanschoren/status/1072429608014897153)

[Meta Learning Taxonomy](http://metalearning-symposium.ml/files/vinyals.pdf)

---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

