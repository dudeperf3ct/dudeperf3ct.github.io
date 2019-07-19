---
layout:     post
title:      Fun in Deep Learning Project
date:       2019-05-17 12:00:00
summary:    This post will provide a 
categories: project
published : false
---


# Fun in Deep Learning Project

In this post, we will create a project, a text recognizer application. Here we will detail all the process that takes place in creating a deep learning project and various practices. This post will be updated multiple times as we will deal with a lot of experiments spanning multiple weeks. 

> All the codes safely stored in github repo [Keras]()

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.

<p align="center">
<img src='/images/dl_project/deep_learning_project.jpg' width="50%"/> 
</p>



Feel free to jump anywhere,

- [](#)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)


Full Stack Deep Learning Bootcamp provides an excellent guide on many different questions keeping up DL practitioner such as, "How to " As the graphics below describes, this course was really about practices in creating production-ready projects.

<p align="center">
<img src='/images/dl_project/course.png' width="50%"/> 
</p>


We will also look at an amazing case-study done at Dropbox

We will steal some of the slides from the lectures that serve as an excellent to-do's as a starting point of any DL project. We will apply all these ideas and answer all the questions in context of our project.

We start with outlining these steps that are performed for every project in Deep Learning.

<p align="center">
<img src='/images/dl_project/steps.png' width="50%"/> 
</p>

1. Planning & Project Setup 

In this first step, we lay out what are the goals of the project. We determine what the requirement the project needs and make sure have enough resources allocated for the project.

2. Data collection & labelling

Here comes the most critical step, we look for different ways in which we can collect appropriate data for our project and look for cheap ways to label the data. Here we focus on the questions, How hard is to get data?, How expensive is data labelling? or How much data will be needed?

3. Training & Debugging

In this step, we start with implementing baselines. We look for any SoTA models and reproduce those. All we need is to make our model more robust and effective. We also look at what metric do we care about and decide which metric to the model optimize for.

4. Deploying & Testing

In last step, we write test functions to test out the model in some version control (not after deploying in real-world but before) to check the robustness of the model and once happy with the results we are ready to deploy.

Notice the flow is not linear or sequential, there is a lot of backtracking and improving as we improve our beliefs about the project. One example could be that having decided goal and collected data, we moved on to training step but once there we realize that our labels are unreliable or realize that goal is too hard and thus we backtrack to second or first step from third step.

<p align="center">
<img src='/images/dl_project/more_steps.png' width="50%"/> 
</p>



<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology



---

# Further Reading


---

# Footnotes and Credits

[Meme](https://hackernoon.com/latest-deep-learning-ocr-with-keras-and-supervisely-in-15-minutes-34aecd630ed8)

[Steps and detailed steps](https://full-stack-deep-learning.aerobaticapp.com/e372_52326459-3750-4663-b795-e78e05f84f0c/assets/slides/fsdl_2_projects.pdf)

---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

