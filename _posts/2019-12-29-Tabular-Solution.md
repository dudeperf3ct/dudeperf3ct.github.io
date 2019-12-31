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
  - [Bellman Expectation Equation](#bellman-expectation-equation)
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

A Markov decision process (MDP) is a Markov reward process with decisions. Markov decision processes(MDP) are used to describe an environment in reinforcement learning. In MDPs, we are also concerned with selecting different action associated with every state. The environment responds with a new state and reward for choosing a particular action when in a given particular state. Almost all RL problems can be formalised as MDPs.

Markov Decision Process is a tuple ($$\mathcal{S}$$, $$\mathcal{A}$$,, $$\mathcal{P}$$, $$\mathcal{R}$$, $$\gamma$$),
- $$\mathcal{S}$$ is a (finte) set of states
- $$\mathcal{A}$$ is a (finte) set of actions
- $$\mathcal{P}$$ is a state transition probability matrix, $$\mathcal{P}^{a}_{ss^{'}}$$ = $$\mathbb{P}[S_{t+1} = s^{'} \vert S_{t} = s, A_{t} = a]$$
- $$\mathcal{R}$$ is a reward function, $$\mathcal{R}_{s}$$ = $$\mathbb{E}[\mathcal{R}^{a}_{t+1} \vert S_{t} = s, A_{t} = a]$$
- $$\gamma$$ is a discount factor, $$\gamma$$ $$\in$$ [0, 1]

### Bellman Expectation Equation

We use bellman equation to show how current state is related to successive state for both value functions. We can apply this recusrive equation for each sequence in each episode of an episodic task.

- Return

In RL, we seek to maximise the expected return where the return $$G_{t}$$ is the total discounted reward from time-step $$t$$. For episodic tasks, $$G_{t} = R_{t+1} + R_{t+2} ... + R_{T}$$, where T is the terminal state. For continuous tasks, $$G_{t} = R_{t+1} + \gamma * R_{t+2} ... + \gamma^{2} * R{t+3} = \sum_{k=0}^{\inf} \gamma^{k} R_{t+k+1} $$, where $$\gamma$$ is the discount rate. 

The recursive equation of relating return at current time step $$t$$ to next time step $$t+1$$ is given by,

$$
\begin{aligned}
G_{t} = R_{t+1} + \gamma * G_{t+1}
\end{aligned}
$$


- Value Functions

Almost all reinforcement learning algorithms involve estimating value functions. Value functions determine how good is it to be in a particular state (state-value function) or how good is to take a particular action in given state (action-value function). The state-value and action-value are related by the following equation, 

$$
\begin{aligned}
v_{\pi}(s) &= \sum_{a \in A}\pi(a \vert s)q_{\pi}(s, a)
\end{aligned}
$$

State-value function

The state-value function of an MDP is expected return starting from state $$s$$, and then following policy $$\pi$$,

$$
\begin{aligned}
v_{\pi}(s) &= \mathbb{E}_{\pi}[G_{t} \vert S_{t} = s]\\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \vert S_{t} = s]\\
&= \sum_{a \in A}\pi(a \vert s)\mathcal{R}_{s}^{a} + \gamma \sum_{s^{'} \in S}\mathcal{P}_{ss^{'}}^{a}v_{\pi}(s^{'})
\end{aligned}
$$

This equation is Bellman equation for $$v_{\pi}$$. I When in state $$s$$, an agent takes an action $$a$$ based on its policy $$\pi$$. The environment could respond with one of several next states $$s^{'}$$, along with immediate reward $$r$$. Bellman equation averages over all the possibilities, weighting each by its probability of occurring. It expresses a relationship between the value of a state and the values of its successor states.

Action-value function

The action-value function of an MDP is expected return starting from state $$s$$, taking action $$a$$ and then following policy $$\pi$$,

$$
\begin{aligned}
q_{\pi}(s, a) &= \mathbb{E}_{\pi}[G_{t} \vert S_{t} = s, A_{t} = a]\\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma q_{\pi}(S_{t+1}, A_{t+1}) \vert S_{t} = s, A_{t} = a]\\
&= \mathcal{R}_{s}^{a} + \gamma \sum_{s^{'} \in S}\mathcal{P}_{ss^{'}}^{a}\sum_{a^{'} \in A}\pi(a^{'} \vert s{'})q_{\pi}(s^{'}, a^{'})
\end{aligned}
$$

This equation is Bellman equation for $$q_{\pi}$$. I When in state $$s$$ and taking an action $$a$$ based on its policy $$\pi$$. The environment could respond with one of several next states $$s^{'}$$, along with immediate reward $$r$$. 


### Bellman Optimality Equation


- Optimal Value Function

The optimal state-value function $$v_{*}(s)$$ is the maximum state-value function over all policies.

$$
\begin{aligned}
v_{*}(s) &= max_{\pi}v_{\pi}(s)\\
&= max_{a}\mathcal{R}_{s}^{a} + \gamma \sum_{s^{'} \in S}\mathcal{P}_{ss^{'}}^{a}v_{\pi}(s^{'})
\end{aligned}
$$

The optimal action-value function $$q_{*}(s, a)$$ is the maximum action-value function over all policies.

$$
\begin{aligned}
q_{*}(s, a) &= max_{\pi}q_{\pi}(s, a)\\
&= \mathcal{R}_{s}^{a} + \gamma \sum_{s^{'} \in S}\mathcal{P}_{ss^{'}}^{a}max_{a^{'}}q_{\pi}(s^{'}, a^{'})
\end{aligned}
$$


- Optimal Policy

A policy is defined to be better than or equal to a policy $$\pi^{'}$$ if its expected return is greater than or equal to than of $$\pi^{'}$$ for all states. There is always at least one policy ($$\pi_{*}$$) that is better than or equal to all other policies ($$\pi_{*} \ge \pi}\forall \pi$$). This is an optimal policy. There can be more than one optimal policies. All optimal policies achieve optimal state-value function ($$v_{\pi_{*}}(s) = v_{*}(s)$$) and action-value function ($$q_{\pi_{*}}(s, a) = q_{*}(s, a)$$).

We can obtain optimal policy directly if we have $$q_{*}(s, a)$$.

$$
\begin{aligned}
\pi_{*}(a \vert s) = 
\begin{cases} 
1 &\mbox{if } a = argmax_{a \in A}q_{*}(s, a)\\
0 & otherwise 
\end{cases}
\end{aligned}
$$


### Backup Diagrams

Backup diagrams are used to present the transitions of states and actions for an agent graphically. We call such diagrams backup diagrams because 


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
