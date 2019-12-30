---
layout:     post
title:      Tabular Solution Methods
date:       2019-12-29 12:00:00
summary:    
categories: rl
published : false

---


# Tabular Solution Methods


<p align="center">
<img src='/images/series_rl/rl_meme.jpg' width="50%"/> 
</p>


Feel free to jump anywhere,

- [Tabular Solution Methods](#tabular-solution-methods)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)


# Terminologies


### Markov Process

In Markov processes, the states captures all relevant information from the past agent–environment interaction. Theses states are said to have Markov property. The states with Markov property are memoryless. For e.g we can predict the next move on chess board given any configuration of the board i.e. all that matter to predict the next move is the current state. It doesn't matter how we got there. The current state is a sufficient statistic of the future.

> The future is independent of the past given the present.

$$
P(S_{t} \vert S_{1}, S_{2}, ..., S_{t-1}) = P(S_{t} \vert S_{t-1})
$$


### Markov Reward Process



### Markov Decision Process

Markov decision processes(MDP) are used to describe an environment in reinforcement learning. Almost all RL problems can be formalised as MDPs.

### Value function




# Tabular Solution Methods

Tabular Solutions are preferred method for solving RL problems when state and action space is small. The state functions and action-state functions are represented as tables. For such problems, exact optimal policy and optimal value functions can be found. 


<span class='orange'>Happy Learning!</span>


# Further Reading

Reinforcement Learning An Introduction 2nd edition : [Chapter 1](http://incompleteideas.net/sutton/book/RLbook2018.pdf)

UCL RL Course by David Silver : [Lecture 1](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=1)

UC Berkeley CS285 Deep Reinforcement Learning : [Lecture 1](https://www.youtube.com/watch?v=SinprXg2hUA&list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&index=2&t=0s)

---

# Footnotes and Credits



---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)

---
