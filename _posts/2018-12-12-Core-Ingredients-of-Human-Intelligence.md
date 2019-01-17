---
layout:     post
title:      Core Ingredients of Human Intelligence
date:       2018-12-12 12:00:00
summary:    Transform your plain text into static websites and blogs. Simple, static, and blog-aware.
categories: human intelligence ingredients
published : false
---


## Introduction

The [first part](https://dudeperf3ct.github.io/agi/paper-review/2018/11/09/Buliding-machines-that-learn-and-think-like-people/) of this series of blog introduces the problem for human-like intelligence.

The authors of the paper "Building machines that learn and think like people" propose 3 core ingredients needed for human-like learning and thought. Authors believe that integrating them will produce significantly more powerful and more human-like learning and thinking abilities than we currently see in AI systems.

In second post, we will go through each of the ingredients proposed by the authors and see how each of these ingredients helps in solving the two challenges of Character Challenge and Frostbite Challenge. 

The 3 core ingredients are:
 
 1. [Developmental start-up software](#1-developmental-start-up-software)
    - [Intuitive Physics](#11-intuitive-physics)
    - [Intuitive Psychology](#12-intuitive-psychology)
 2. [Learning as rapid model building](#2-learning-as-rapid-model-building)
    - [Compositionality](#21-compositionality)
    - [Causality](#22-causality)
    - [Learning-to-learn](#23-learning-to-learn)
 3. [Thinking Fast](#3-thinking-fast) 
    - [Approximate inference in structured models](#31-approximate-inference-in-structured-models)
    - [Model-based and model-free reinforcement learning](#32-model-based-and-model-free-reinforcement-learning)


## 3 Core Ingredients

### 1. Developmental start-up software

-babies_think.jpeg

> In TED Talk by Alison Gopnik on "[What do babies think?]"(https://www.ted.com/talks/alison_gopnik_what_do_babies_think), "What is going in this baby's mind? If you'd asked people this 30 years ago, most people, including psychologists, would have said that this baby was irrational, illogical, egocentric -- that he couldn't take the perspective of another person or understand the cause and effect. In last 20 years, developmental science has completely overturned that picture. So, in some ways, we think that this baby's thinking is like the thinking of the most brilliant scientists."

Babies and young children are like scientists. Scientists do stastical analysis while babies and young children do experiements and draw conclusions. Grown-ups think in terms of a goal — planning, acting and doing to make things happen or accomplish the goal. Babies don't have that narrow, goal-directed approach to the world. They're open to all the information that will tell them something new. 

> The "child as scientist" proposal further views the process of learning itself as also scientist-like, with recent experiments showing the children seek out data to distinguish between hypothese, isolate variables, test causal hypotheses, make use of the data-generating process in drawing conclusions, and learn selectively from others.

One such study done by Fei Xu at University of California, Berkeley, shows how even babies can understand the relation between a statistical sample and a population. 8-month old babies were shown box full of mixed ping-pong balls: for instance, 80% white and 20% red. The experimenter would then take out 5 balls, at random. The babies were more surprised (that is, looked longer and more intently at the scene) when the experimenter pulled four red balls and one white one out of the box -- an improbable outcome -- than when she pulled out four white balls and one red one.


#### 1.1 Intuitive physics

Researchers such as Renée Baillargeon of the University of Illinois and Elizabeth S. Spelke of Harvard University found that infants understand fundamental physical relations such as movement trajectories, gravity and containment. They look longer at a toy car appearing to pass through a solid wall than at events that fit basic principles of everyday physics. By the time they are three or four, children have elementary ideas about biology and a first understanding of growth, inheritance and illness. This early biological understanding reveals that children go beyond superficial perceptual appearances when they reason about objects. 

child_learning.gif

In the case of our challenges, learning to play Frostbite, incorporating a physics-enginer-based representation could help DQNs learn to play games such as Frostbite in a faster and more general way, whether the phyiscs knowledge is capture implicitly in a neural network or more explicity in a simulator. It can also reduce the need of larger datasets and retraining if objects like birds, fish,e etc are slightly modified in their behavior, reward structure or apearance. For e.g. when a new object type such as a bear is introduced, in later levels of Frostbite, a network endowed with intuitive physics would also have an easier time adding this object type to its knowledge.


#### 1.2 Intuitive psychology



### 2. Learning as rapid model building



#### 2.1 Compositionality


