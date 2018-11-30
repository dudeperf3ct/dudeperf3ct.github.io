---
layout:     post
title:      Core Ingredients of Human Intelligence
date:       2018-11-20 12:00:00
summary:    Transform your plain text into static websites and blogs. Simple, static, and blog-aware.
categories: human intelligence ingredients
published : false
---


## Introduction

The first part of this series of blog introduces the problem for human-like intelligence.

The authors of the paper "Building machines that learn and think like people" propose 3 core ingredients needed for human-like learning and thought. Authors believe that integrating them will produce significantly more powerful and more human-like learning and thinking abilities than we currently see in AI systems.

In this post, we will go through each of the ingredients proposed by the authors.

The 3 core ingredients are:
 
 1. [Developmental start-up software](#11developmental-start-up-software)
    - Intuitive Physics
    - Intuitive Psychology
 2. Learning as rapid model building
    - Compositionality
    - Causality
    - Learning-to-learn
 3. Thinking Fast 
    - Approximate inference in structured models
    - Model-based and model-free reinforcement learning


## 3 Core Ingredients

### 1. Developmental startup-software

Human excel at number of domains including number (numerical and set operations), space (geometry and navigation), physics (inanimate objects and mechanics), and psychology (agents and groups). All of these domains may be important augmentations to current machine learning.

> In TED Talk by Alison Gopnik, "What is going in this baby's mind? If you'd asked people this 30 years ago, most people, including psychologists, would have said that this baby was irrational, illogical, egocentric -- that he couldn't take the perspective of another person or understand the cause and effect. In last 20 years, developmental science has completely overturned that picture. So, in some ways, we think that this baby's thinking is like the thinking of the most brilliant scientists."

The "child as scientist" proposal further views the process of learning itself as also scientist-like, with recent experiments showing the children seek out data to distinguish between hypothese, isolate variables, test causal hypotheses, make use of the data-generating process in drawing conclusions, and learn selectively from others.

#### 1.1 Intuitive physics

Young children have a rich knowledge of intuitive physics. 

<div>
  <p align='center'>
 <img src="/images/aww.png" alt="Awwwww" width="80%" />
  </p>
  <p align='center'><span class='gray'>If you push this truck off the platform and if it doesn't fall and below the age of 2-6 months, babies don't care. They say sure, yeah, why not. After 6 months, they say (show by expression) this can't be, this truck has to fall.</span>
  </p>
</div>

At the age 2 months, and possibly earlier, human infants expect inanimate objects to follow principles of persistence, continuity, cohesion and solidity. These expectations guide object segmentation in early infancy, emerging before apperance-based cues such as color, texture, and perceptual goodness. These expectations also go on to guide later learning. 

<p align='center'>
 <img src="/images/early_acquistion_learning.png" alt="Early phases of learning"  width="80%"/>
</p>

At around 6 months, infants have already developed different expectations of rigid bodies, soft bodies, and liquids. By their first birthday, infants have gone through several transitions of comprehending basic physical concepts such as intertia, support, containment, and collisions. Various methods ranging from decision trees, to cues, to list of rules have tried to model these early physical principles and concepts but there exists "No Free Lunch" theorem.

Consider example of, a physics-engine reconstruction of a tower of wodden blocks from the game Jenga can be used to predict whether and how a tower will fall, finding close quantitaive fits of how adults and infants make these predictions.


<p align='center'>
 <img src="/images/IPE.png" alt="Intuitive Physics Engine"  width="80%"/>
</p>

The above simulator-engine provides intuitive physics engine approach to scene understanding where engine takes input through perception, language, memory and other faculties. It then constructs a physical scenewith obects, physical properties, and forces simulates the scene’s development over time and hands the output to other reasoning systems and hands the output to other reasoning systems. Many possible tweaks to input can result in very different scenes, requiring the potential discovering, training, and evaluation of new features for each tweak. Facebook AI researchers trained deep CNN system (PhysNet), to predict the stability of block towers from simulated images as above image with simpler configurations of two, three or four cubical blocks stacked vertically. PhysNet acheived impressive results by generalizing to simple real images of block towers, matching human performance on these images, meanwhile exceeding human performance on synthetic images. One limitation of PhysNet is that it requires extensive training - 100,000 and 200,000 scenes - to learn judgements for just a single task (will tower fall?) on a narrow range of scenes (towers with two to four cubes). In contrast, people require far less experience to perform any particular task, and can generalize to many novel judgements and complex scenes with no new training required. One can argue that humans have a large amount of physics experience through interacting with the world more generally. 

Consider for example, a network that learns to predict the trajectories of several balls bouncing in a box. If this network has actually learned something like Newtonian mechanices, then it should be able to generalize to interestingly different scenarios - at a minimum different numbers of differently shaped objects, bouncing in boxes of different shapes and sizes and orientations with respect to gravity. As deep CNN has feature learning hierarchy, low level features like edges, textures are learned in initial layers and gradually high level features identify full objects, similarly if deep networks trained with physics-realted data, it remains to be seen whether higher layers will encode objects, general physical properties, forces and approximately Newtonian dynamics. The transfer learning of these deep networks will help transferring these learned features to many different tasks similar to transfer learning in deep CNN networks. It may be difficult to integrate object and physics-based primitives into deep neural networks, but payoff in terms of learning speed and performance could be great for many tasks. 

Incorporating a physics-engine-based representation could help DQNs learn to play games like frostbire in a faster and more general way, whether physics knowledge is captured implicitly in a neural network or more explicitly in a simulator. Beyond reducing the amount of training data and potentially improving the level of performance reached by the DQN, it could eliminate the need to retrain a Frostbite network if the objects (e.g., birds, ice-floes and fish) are slightly altered in their behavior, reward-structure, or appearance.  When a new object type such as a bear is introduced, as in the later levels of Frostbite, a network endowed with intuitive physics would also have an easier time adding this object type to its knowledge.

<span class='red'>Could neural networks be trained to emulate a general-purpose physics simulator, give the right type and quantity of training data, such as the raw input experienced by child?</span>

#### 1.2 Intuitive psychology

<span class='red'>In Intuitive psychology important influence is on human learning and thought.</span>

Pre-verbal infants distinguish among the animate agents and inanimate objects. This can be innate or early-present detectors for low-level cues, such as presence of eyes, motion intiated from rest or infants expects agents to act contingently and reciprocally, to have goals, and to take efficient actions towards those goals subject to constraints. Also, infants expect agent to act in a goal-directed, efficient and socially sensitive fashion.


Consider for example a scenario in which an agent A is moving towards a box, and an agent B moves in a way that blocks A from reaching the box. Infants and adults are likely to interpret B’s behavior as ‘hindering’.

This inference could be captured by a cue that states if an agent’s expected trajectory is prevented from completion, the
blocking agent is given some negative association.

In Frostbite challenge, we discussed how people can learn to play the game extremely quickly by watching an experienced player for just a few minutes and then playing a few rounds themselves. Intuitive psychology provides a basis for efficient learning from others, especially in teaching settings with the goal of communicating knowledge efficiently. In the case of watching an expert play Frostbite, whether or not there is an explicit goal to teach, intuitive psychology lets us infer the beliefs, desires, and intentions of the experienced player.  For instance, we can learn that the birds are to be avoided from seeing how
the experienced player appears to avoid them.  We do not need to experience a single example of encountering a bird – and watching the Frostbite Bailey die because of the bird – in order to infer that birds are probably dangerous. It is enough to see that the experienced player’s avoidance behavior is best explained as acting under that belief.

There are several ways that intuitive psychology could be incorporated into contemporary deep learning systems. While the origins of intuitive psychology is still a matter of debate, it is clear that these abilities are early-emerging and play an important role in human learning and thought, as exemplified in the Frostbite challenge and when learning to play novel video games more broadly.

### 2. Learning as rapid model building

