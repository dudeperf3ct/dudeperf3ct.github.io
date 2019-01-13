---
layout:     post
title:      Core Ingredients of Human Intelligence
date:       2018-12-12 12:00:00
summary:    Transform your plain text into static websites and blogs. Simple, static, and blog-aware.
categories: human intelligence ingredients
published : false
---


## Introduction

The first part of this series of blog introduces the problem for human-like intelligence.

The authors of the paper "Building machines that learn and think like people" propose 3 core ingredients needed for human-like learning and thought. Authors believe that integrating them will produce significantly more powerful and more human-like learning and thinking abilities than we currently see in AI systems.

In this post, we will go through each of the ingredients proposed by the authors and see how each of these ingredients helps in solving the two challenges of Character Challenge and Frostbite Challenge. 

The 3 core ingredients are:
 
 1. [Developmental start-up software](#1-developmental-start-up-software)
    - [Intuitive Physics](#11-intuitive-physics)
    - [Intuitive Psychology](#12-intuitive-psychology)
 2. [Learning as rapid model building](#2-learning-as-rapid-model-building)
    - [Compositionality](#21-compositionality)
    - Causality
    - Learning-to-learn
 3. Thinking Fast 
    - Approximate inference in structured models
    - Model-based and model-free reinforcement learning


## 3 Core Ingredients

### 1. Developmental start-up software

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

<p align="center">
<a href="http://www.youtube.com/watch?v=Z-eU5xZW7cU">
<img src="http://img.youtube.com/vi/Z-eU5xZW7cU/0.jpg" />
</a>
<p align="center"><span class='gray'> <b>Video showing intuitive psychology add footnote</b> </span></p>
</p>

Consider for example a scenario in which an agent A is moving towards a box, and an agent B moves in a way that blocks A from reaching the box. Infants and adults are likely to interpret B’s behavior as ‘hindering’.

This inference could be captured by a cue that states if an agent’s expected trajectory is prevented from completion, the
blocking agent is given some negative association.

In Frostbite challenge, we discussed how people can learn to play the game extremely quickly by watching an experienced player for just a few minutes and then playing a few rounds themselves. Intuitive psychology provides a basis for efficient learning from others, especially in teaching settings with the goal of communicating knowledge efficiently. In the case of watching an expert play Frostbite, whether or not there is an explicit goal to teach, intuitive psychology lets us infer the beliefs, desires, and intentions of the experienced player.  For instance, we can learn that the birds are to be avoided from seeing how
the experienced player appears to avoid them.  We do not need to experience a single example of encountering a bird – and watching the Frostbite Bailey die because of the bird – in order to infer that birds are probably dangerous. It is enough to see that the experienced player’s avoidance behavior is best explained as acting under that belief.

There are several ways that intuitive psychology could be incorporated into contemporary deep learning systems. While the origins of intuitive psychology is still a matter of debate, it is clear that these abilities are early-emerging and play an important role in human learning and thought, as exemplified in the Frostbite challenge and when learning to play novel video games more broadly.

### 2. Learning as rapid model building


In recent years, machine learning has found particular success using backpropagation and large data sets to solve difficult pattern recognition problems.  While these algorithms have reached human-level performance on several challenging benchmarks, they are still far from matching human-level learning in other ways.  Deep neural networks often need more data than people do in order to solve the same types of problems, whether it is learning to recognize a new type of object or learning to play a new game.

Children may only need to see a few examples of the concepts hairbrush, pineapple or lightsaber before they largely ‘get it,’ grasping the boundary of the infinite set that defines each concept from the infinite set of all possible objects. Children are far more practiced than adults at learning new concepts – learning roughly nine or ten new words each day after beginning to speak through the end of high school – yet the ability for rapid “one-shot” learning does not disappear in adulthood.  An adult may need to see a single image or movie of a novel two-wheeled vehicle to infer the boundary between this concept and others, allowing him or her to discriminate new examples of that concept from similar looking objects of a different type.

In the context of learning new handwritten  characters or learning to play Frostbite, the MNIST benchmark includes 6000 examples of each handwritten digit, and the DQN of V.Mnih et al. played each Atari video game for approximately 924 hours of unique training experience. In both cases, the algorithms are clearly using information less efficiently than a person learning to perform the same tasks.

Even with just a few examples, people can learn remarkably rich conceptual models. One indicator of richness is the variety of functions that these models support like prediction, action, communication, imagination, explaination, and composition. These abilities are not independent; rather they hang together and interact, coming for free with the acquisition of the underlying concept. In the context of Frostbite, a learner who has acquired the basics of the game could flexibly apply their knowledge to an infinite set of Frostbite variants. The acquired knowledge supports reconfiguration to new tasks and new demands, such as modifying the goals of the game to survive while acquiring as few points as possible, or to efficiently teach the rules to a friend.

What additional ingredients may be needed in order to rapidly learn more powerful and more general-purpose representations?

#### 2.1 Compositionality

Compositionality is the classic idea that new representations can be constructed through the combination of primitive elements.  In computer programming, primitive functions can be combined together to create new functions, and these new functions can be further combined to create even more complex functions. This function hierarchy provides an efficient description of higher-level functions, like a part hierarchy for describing complex objects or scenes. Compositionality is also at the core of productivity:  an infinite number of representations can be constructed from a finite set of primitives, just as the mind can think an infinite
number of thoughts, utter or understand an infinite number of sentences, or learn new concepts from a seemingly infinite space of possibilities.

Handwritten characters are inherently compositional, where the parts are pen strokes and relations describe how these strokes connect to each other.

An efficient representation for Frostbite should be similarly compositional and productive. A scene from the game is a composition of various object types, including birds, fish, ice floes, igloos,etc. Representing this compositional structure explicitly is both more economical and better for generalization, as noted in previous work on object-oriented reinforcement learning. Many repetitions of the same objects are present at different locations in the scene, and thus representing each as an identical instance of the same object with the same properties is important for efficient representation and quick learning of the game. Further, new levels may contain different numbers and combinations  of objects, where a compositional representation of objects – using intuitive physics and intuitive psychology as glue – would aid in making these crucial generalizations.

We look forward to seeing these new ideas continue to develop, potentially providing even richer notions of compositionality in deep neural networks that lead to faster and more flexible learning. To capture the full extent of the mind’s compositionality, a model must include explicit representations of objects, identity, and relations – all while maintaining a notion of “coherence”  when understanding novel configurations. Coherence is related to our next principle, causality, which is discussed in the section that follows.

