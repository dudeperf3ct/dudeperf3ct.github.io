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

- [Terminologies](#terminologies)
  - [Markov Process](#markov-process)
  - [Markov Reward Process](#markov-reward-process)
  - [Markov Decision Process](#markov-decision-process)
  - [Bellman Equation](#bellman-equation)
  - [Bellman Optimality Equation](#bellman-optimiality-equation)  
- [Tabular Solution Methods](#tabular-solution-methods)
  - [Dynamic Programming](#dynamic-programming)
  - [Monte-Carlo](#monte-carlo)
  - [TD Learning](#td-learning)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)

# Terminologies


### Markov Process

In Markov processes, the states captures all relevant information from the past agent–environment interaction. These states are said to have Markov property. The states with Markov property are memoryless. For e.g we can predict the next move on chess board given any configuration of the board i.e. all that matter to predict the next move is the current state. It doesn't matter how we got there. The current state is a sufficient statistic of the future.

> The future is independent of the past given the present.

$$
\begin{aligned}
P(S_{t} \vert S_{1}, S_{2}, ..., S_{t-1}) = P(S_{t} \vert S_{t-1})
\end{aligned}
$$

Markov Process(or Markov Chain) is a tuple ($$\mathcal{S}$$, $$\mathcal{P}$$),
- $$\mathcal{S}$$ is a (finte) set of states
- $$\mathcal{P}$$ is a state transition probability matrix, $$\mathcal{P}_{ss^{'}}$$ = $$\mathbb{P}[S_{t+1} = s^{'} \vert S_{t} = s]$$


### Markov Reward Process

A Markov reward process is a Markov chain with values. In Markov reward processes, each transition is associated with a reward. The agent-environment interaction can be episodic i.e. broken into episodes, terminating after ending up in a terminal state or continuous, in which the interaction does not naturally break into episodes but continues without limit. That is why we introduce a discounted delayed reward. If $$\gamma$$ = 0, we get a myopic agent concerned only with maximizing immediate rewards and $$\gamma$$ = 1, we get a far-sighted agent which takes future rewards into account more strongly.

Markov Reward Process is a tuple ($$\mathcal{S}$$, $$\mathcal{P}$$, $$\mathcal{R}$$, $$\gamma$$),
- $$\mathcal{S}$$ is a (finte) set of states
- $$\mathcal{P}$$ is a state transition probability matrix, $$\mathcal{P}_{ss^{'}}$$ = $$\mathbb{P}[S_{t+1} = s^{'} \vert S_{t} = s]$$
- $$\mathcal{R}$$ is a reward function, $$\mathcal{R}_{s}$$ = $$\mathbb{E}[\mathcal{R}_{t+1} \vert S_{t} = s]$$
- $$\gamma$$ is a discount factor, $$\gamma$$ $$\in$$ [0, 1]

### Markov Decision Process

A Markov decision process (MDP) is a Markov reward process with decisions. Markov decision processes(MDP) are used to describe an environment in reinforcement learning. In MDPs, we are  
Almost all RL problems can be formalised as MDPs.
Markov Decision Process is a tuple ($$\mathcal{S}$$, $$\mathcal{A}$$,, $$\mathcal{P}$$, $$\mathcal{R}$$, $$\gamma$$),
- $$\mathcal{S}$$ is a (finte) set of states
- $$\mathcal{A}$$ is a (finte) set of actions
- $$\mathcal{P}$$ is a state transition probability matrix, $$\mathcal{P}^{a}_{ss^{'}}$$ = $$\mathbb{P}[S_{t+1} = s^{'} \vert S_{t} = s, A_{t} = a]$$
- $$\mathcal{R}$$ is a reward function, $$\mathcal{R}_{s}$$ = $$\mathbb{E}[\mathcal{R}^{a}_{t+1} \vert S_{t} = s, A_{t} = a]$$
- $$\gamma$$ is a discount factor, $$\gamma$$ $$\in$$ [0, 1]


### Belman Equation

- Return

The return $$G_{t}$$ is the total discounted reward from time-step $$t$$.


- Value Functions


### Belman Optimality Equation

- Optimal Value Function

- Optimal Policy


# Tabular Solution Methods

Tabular Solutions are preferred method for solving RL problems when state and action space is small. The state functions and action-state functions are represented as tables. For such problems, exact optimal policy and optimal value functions can be found. 


### Dynamic Programming


### Monte-Carlo 

### TD-Learning


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
