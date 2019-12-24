---
layout:     post
title:      Series on Reinforcement Learning
date:       2019-12-19 12:00:00
summary:    This post will provide an introduction to reinforcement learning and will outline the topics we will cover in subsequent post.
categories: rl
published : false
---


# Series on Reinforcement Learning


Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.


<p align="center">
<img src='/images/series_rl/rl_meme.jpg' width="50%"/> 
</p>


Feel free to jump anywhere,

- [Reinforcement Learning](reinforcement-learning)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)


# Reinforcement Learning

Max Tegmark in Life 3.0 explains how the origin of goal-oriented behaviour can be traced all the way back to laws of physics, which appear to endow particles with the goal of arranging themselves so as to extract energy from their environment as efficiently as possible. In a way, that is how life evolved, particular arrangement of particles getting so good at copying itself.



Sutton and Barto define reinforcement learning as a computational approach to understanding and automating goal-directed learning and decision making.


There are 4 main subelements of a reinforcement learning system:

- Policy

A policy is a mapping from perceived states of the environment to actions to be taken when in those states. It acts like a lookup table for a agent, directing it to take a particular action when in an particular state. Policies may be stochastic, i.e. specifying probabilities for each action. The policy is the core of a reinforcement learning agent in the sense that it alone is sufficient to determine behavior. It is represented by symbol $$\pi$$.

- Rewards

A reward signal defines the goal of a reinforcement learning problem. Agents can recieve rewards either be recieved at the end of the episode or after taking each step. For example, a chess game will end in reward of +1(win), 0(draw) and -1(lose). Rewards acts as a feedback for agent and thus defines what are the good and bad events for the agent. Reward signals may be stochastic functions of the state of the environment and the actions taken. The agent’s sole objective is to maximize the total reward it receives over the long run. It is represented by symbol $$\mathcal{R}$$.

- Value Function

The value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state. It tells the agent how good it is to be in a particular state. Based on this value functions, agents decides which actions should be preferred. The difference between rewards and value function is rewards are given directly by environment but value function is estimated and re-estimated from the sequences of observation an agent makes over its entire lifetime. It is represented by symbol $$v_{\pi}(s)$$, value of state s under policy $$\pi$$.




<span class='orange'>Happy Learning!</span>


# Further Reading

Reinforcement Learning An Introduction 2nd edition [Chapter 1]()

---

# Footnotes and Credits

[Rewards Meme](https://himanshusahni.github.io/2018/02/23/reinforcement-learning-never-worked.html)



---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

