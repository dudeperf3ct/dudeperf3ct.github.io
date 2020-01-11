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

Almost all reinforcement learning algorithms involve estimating value functions. Value functions determine how good is it to be in a particular state (state-value function) or how good is to take a particular action in given state (action-value function). The state-value function and action-value function are related by the following equation, 

$$
\begin{aligned}
v_{\pi}(s) &= \sum_{a \in A}\pi(a \vert s)q_{\pi}(s, a)
\end{aligned}
$$

**State-value function**

The state-value function of an MDP is expected return starting from state $$s$$, and then following policy $$\pi$$,

$$
\begin{aligned}
v_{\pi}(s) &= \mathbb{E}_{\pi}[G_{t} \vert S_{t} = s]\\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \vert S_{t} = s]\\
&= \sum_{a \in \mathcal{A}}\pi(a \vert s)\mathcal{R}_{s}^{a} + \gamma \sum_{s^{'} \in \mathcal{S}}\mathcal{P}_{ss^{'}}^{a}v_{\pi}(s^{'})
\end{aligned}
$$

This equation is Bellman equation for $$v_{\pi}$$. I When in state $$s$$, an agent takes an action $$a$$ based on its policy $$\pi$$. The environment could respond with one of several next states $$s^{'}$$, along with immediate reward $$r$$. Bellman equation averages over all the possibilities, weighting each by its probability of occurring. It expresses a relationship between the value of a state and the values of its successor states.

**Action-value function**

The action-value function of an MDP is expected return starting from state $$s$$, taking action $$a$$ and then following policy $$\pi$$,

$$
\begin{aligned}
q_{\pi}(s, a) &= \mathbb{E}_{\pi}[G_{t} \vert S_{t} = s, A_{t} = a]\\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma q_{\pi}(S_{t+1}, A_{t+1}) \vert S_{t} = s, A_{t} = a]\\
&= \mathcal{R}_{s}^{a} + \gamma \sum_{s^{'} \in \mathcal{S}}\mathcal{P}_{ss^{'}}^{a}\sum_{a^{'} \in \mathcal{A}}\pi(a^{'} \vert s{'})q_{\pi}(s^{'}, a^{'})
\end{aligned}
$$

This equation is Bellman equation for $$q_{\pi}$$. I When in state $$s$$ and taking an action $$a$$ based on its policy $$\pi$$. The environment could respond with one of several next states $$s^{'}$$, along with immediate reward $$r$$. 


### Bellman Optimality Equation



- Optimal Value Function

The optimal state-value function $$v_{*}(s)$$ is the maximum state-value function over all policies.

$$
\begin{aligned}
v_{*}(s) &= max_{\pi}v_{\pi}(s)\\
&= max_{a \in \mathcal{A}}[\mathcal{R}_{s}^{a} + \gamma \sum_{s^{'} \in \mathcal{S}}\mathcal{P}_{ss^{'}}^{a}v_{\pi}(s^{'})]
\end{aligned}
$$

The optimal action-value function $$q_{*}(s, a)$$ is the maximum action-value function over all policies.

$$
\begin{aligned}
q_{*}(s, a) &= max_{\pi}q_{\pi}(s, a)\\
&= \mathcal{R}_{s}^{a} + \gamma \sum_{s^{'} \in \mathcal{S}}\mathcal{P}_{ss^{'}}^{a}max_{a^{'}}q_{\pi}(s^{'}, a^{'})
\end{aligned}
$$


- Optimal Policy

A policy is defined to be better than or equal to a policy $$\pi^{'}$$ if its expected return is greater than or equal to than of $$\pi^{'}$$ for all states. There is always at least one policy ($$\pi_{*}$$) that is better than or equal to all other policies ($$\pi_{*} \ge \pi}\forall \pi$$). This is an optimal policy. There can be more than one optimal policies. All optimal policies achieve optimal state-value function ($$v_{\pi_{*}}(s) = v_{*}(s)$$) and action-value function ($$q_{\pi_{*}}(s, a) = q_{*}(s, a)$$).

We can obtain optimal policy directly if we have $$q_{*}(s, a)$$.

$$
\begin{aligned}
\pi_{*}(a \vert s) = 
\begin{cases} 
1 &\mbox{if } a = argmax_{a \in \mathcal{A}}q_{*}(s, a)\\
0 & otherwise 
\end{cases}
\end{aligned}
$$


### Backup Diagrams

Backup diagrams are used to present the transitions of states and actions for an agent graphically. We call such diagrams backup diagrams because we are updating the state(or action) values for current state using the next state(or action). It's like we are updating the information backwards from next state to current state.

We can represent bellman expectation equation using backup diagram shown below and they provide a simple picture as to what the equation means.


The backup diagrams use to represent bellman optimality equation are shown below.


# Tabular Solution Methods

Tabular Solutions are preferred method for solving RL problems when state and action space is small. The state functions and action-state functions are represented as tables. For such problems, exact optimal policy and optimal value functions can be found. 

There are two ways of solving RL problem either using model-based method or model-free method. Model-based methods require a full knowledge of MDP, we are given an MDP ($$\mathcal{S}$$, $$\mathcal{A}$$,, $$\mathcal{P}$$, $$\mathcal{R}$$, $$\gamma$$). On other hand, model-free methods do not require full knowledge of MDP, given a policy $$\pi$$ and series of episodes, we use the experience to solve RL prediction and control problem.

## Model-based methods

The goal in model-based learning methods is given an MDP and policy, either evaluate a given policy (prediction problem), finding expected returns for the states or finding an optimal policy for given MDP (control problem).

### Dynamic Programming

Dynamic programming is about breaking the overall goal into sub-goal and solving sub-goal optimally. In DP, we are given given a perfect model of the environment as a MDP. We know the complete dynamics of the environment i.e. if I am in a given state, what all possible actions I can take? After taking an action, environment sends us to one possible state of all next states (depending on transition probability). The prediction problem involves evaluating a policy for a given MDP and a policy. (How good is this policy?) We use policy evaluation method to evaluate given policy. The control problem involves solving an MDP, finding an optimal policy. (What is the best policy for given MDP?) We use either policy iteration or value iteration methods to find an optimal policy (or optimal value function).

- Policy Evaluation

In policy evaluation, given a MDP and policy we evaluate a policy by updating value function of states iteratively until convergence i.e, we apply Bellman expectation equation for state-value function iteratively. We initialize $$v_{1}$$ to be 0 and update value functions $$v_{1},v_{2},...,v_{k}$$ for certain iterations k, such that $$\vert v_{k}-v_{k-1} \vert$$ does not exceed some predefined threshold. 

$$
\begin{aligned}
v_{k+1}(s) &= \sum_{a \in \mathcal{A}}\pi(a \vert s)\mathcal{R}_{s}^{a} + \gamma \sum_{s^{'} \in S}\mathcal{P}_{ss^{'}}^{a}v_{k}(s^{'})
\end{aligned}
$$

- Policy Iteration

Policy iteration consits of 2 steps. Given a policy, we evaluate given policy using policy evaluation from above and we act greedy with respect to value function obtained in policy evaluation step to get a improved policy. This step is called policy improvement step. We repeat these 2 steps until policy converges i.e there is no change in old and new improved policy.

Figure

where E denotes policy evaluation and I denotes policy improvement. This method converges to optimal value function ($$v_{*}$$) and optimal policy ($$\pi_{*}$$) in a finite number of iterations for a finite MDP. At convergence, we statisfy Bellman optimality equation for both policy and value function.


- Value Iteration

In policy iteration, we first evaluate a policy for some iterations and then move on to policy improvement step. What if we evaluate policy for 1 iteration? This will reduce the time we wait for value function to converge in policy evaluation step. This algorithm of policy evaluation for 1 iteration(update of each state) and policy improvement is called value iteration. We can combine the policy improvement and truncated policy evaluation steps in one equation. This equation turns Bellman optimality equation into update rule. This equation guarantees convergence to optimal value function similar to policy iteration. We keep track of policies in value iteration implicitly (take one step and choosing action that maximises expected reward).

$$
\begin{aligned}
v_{k+1}(s) &= max_{a \in \mathcal{A}}[\mathcal{R}_{s}^{a} + \gamma \sum_{s^{'} \in S}\mathcal{P}_{ss^{'}}^{a}v_{k}(s^{'})]\\
\end{aligned}
$$

In value iteration, only a single iteration of policy evaluation is performed in between each policy improvement.

There is another variant of iterative DP algorithms, Asynchronous DP where values of states are updated in any order whatsoever. DP algorithms are not practical for very large state space(or action space) because DP uses full-width backups. The number of deterministic policies for number of actions $$k$$ and states $$n$$ are $$n^{k}$$. DP is exponentially faster than exhaustively searching each possible policy. All the DP algorthims seen above can be summarized in the table below.


| Problem       | Bellman Equation           | Algorithm  |
| ------------- |:-------------:| :-----:|
| Prediction      | Bellman Expectation Equation | IterativePolicy Evaluation |
| Control      | Bellman Expectation Equation + Greedy Policy Improvement     |  Policy Iteration |
| Control | Bellman Optimality Equation      |    Value Iteration |

## Model-free methods

To run model-based methods, we require full knowledge of MDP transitions. In model-free methods, we don't have the complete dynamics of the environment. Hence, we interact with the environment to generate episodes of experience. In model-free methods, the model generates only sample transitions, not complete probability distribution of all possible transitions that is required for DP.

### Monte Carlo 

MC uses *experiences*, sample of sequences of states, actions, and rewards to estimate the average sample returns (*not expected returns as seen in DP*). As more returns are observed, the average should converge to the expected value. MC methods works only for episodic tasks. Each episode contains experiences and each episode eventually terminates. Only on the completion of an episode are value estimates and policies changed. This shows that MC methods are incremental learning methods, episode-by-episode sense but not in a step-by-step (online) sense. In MC like DP, we solve two problems of *prediction* and *control*.  In MC prediction, given a policy we estimate state-value function or action-value function. In MC control, the goal is to find approximate optimal policy for an unknown MDP environment.

- First Visit

As seen in policy evaluation method, we wish to estimate $$v_{\pi}(s)$$, given a set of episodes obtained by following $$\pi$$ and passing through $$s$$. Each occurrence of state $$s$$ in an episode is called a *visit* to $$s$$. With a model, we can estimate policy from values of states (taking one step and choose action that leads to the best combination of reward and next state). In first visit method, we estimate $$v_{\pi}(s)$$ as the average of the returns following *first visits* to $$s$$.

$$
\begin{aligned}
v_{s_{t}} &= v_{s_{t}} + \frac{1}{N(s_{t})}(G_{t} - v_{s_{t}})\\
\end{aligned}
$$

where $$G_{t}$$ is total return from time step $$t$$ and $$N(s_{t})$$ keeps track of visits to state $$s_{t}$$.

If we don't have a model, the state values are not sufficient to provide a policy. In this case, we prefer to evaluate action-value $$q_{\pi}(s, a)$$, the sample returns from taking action $$a$$ from state $$s$$ and following policy $$\pi$$ thereafter. Estimating policy from action values is just taking argmax over the action values, choosing the action with highest action value. In first visit method, we estimate $$q_{\pi}(s, a)$$ as the average of the returns following the first time in each episode that the state $$s$$ was visited and the action $$a$$ was selected.

Another variant of first visit is *every visit*. There can be multiple times state $$s$$ can be visited in the same episode. So, instead of averaging returns of *first visits* to $$s$$, in every visit, we average the returns following all visits to state $$s$$. Both first-visit MC and every-visit MC converge to $$v_{\pi}(s)$$ as the number of visits (or first visits) to s goes to infinity.



### TD-Learning

In DP, all of the estimate values for state where based on the estimates of values of successor states. In RL, this idea is called *bootstrapping*.

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
