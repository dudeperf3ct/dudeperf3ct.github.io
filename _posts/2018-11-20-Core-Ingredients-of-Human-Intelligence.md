---
layout:     post
title:      Core Ingredients of Human Intelligence
date:       2018-11-20 12:00:00
summary:    Transform your plain text into static websites and blogs. Simple, static, and blog-aware.
categories: human intelligence ingredients
published : true
---


## Introduction

The first part of this series of blog introduces the problem for human-like intelligence.

The authors of the paper "Building machines that learn and think like people" propose 3 core ingredients needed for human-like learning and thought. Authors believe that integrating them will produce significantly more powerful and more human-like learning and thinking abilities than we currently see in AI systems.

## 3 Core Ingredients

### 1. Developmental startup-software

Human excel at number of domains including number (numerical and set operations), space (geometry and navigation), physics (inanimate objects and mechanics), and psychology (agents and groups). All of these domains may be important augmentations to current machine learning.

In TED Talk by Alison Gopnik, "What is going in this baby's mind? If you'd asked people this 30 years ago, most people, including psychologists, would have said that this baby was irrational, illogical, egocentric -- that he couldn't take the perspective of another person or understand the cause and effect. In last 20 years, developmental science has completely overturned that picture. So, in some ways, we think that this baby's thinking is like the thinking of the most brilliant scientists."

The "child as scientist" proposal further views the process of learning itself as also scientist-like, with recent experiments showing the children seek out data to distinguish between hypothese, isolate variables, test causal hypotheses, make use of the data-generating process in drawing conclusions, and learn selectively from others.

### 1.1 Intuitive physics

Young children have a rich knowledge of intuitive physics. 

<div>
  <p align='center'>
 <img src="/images/aww.png" alt="Awwwww"/ width="80%">
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

<span class='red'>Could neural networks be trained to emulate a general-purpose physics simulator, give the right type and quantity of training data, such as the raw input experienced by child?</span>
