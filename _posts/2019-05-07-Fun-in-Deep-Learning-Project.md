---
layout:     post
title:      Fun in Deep Learning Project
date:       2019-05-17 12:00:00
summary:    This post will provide a 
categories: project
published : false
---


# Fun in Deep Learning Project

In this post, we will create a project, a text recognizer application. Here we will detail all the process that takes place in creating a deep learning project and various practices. We will also look at an amazing case-study done at Dropbox, [Creating a Modern OCR Pipeline Using Computer Vision and Deep Learning](https://blogs.dropbox.com/tech/2017/04/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning/). This case-study takes us behind the scenes on how the team at Dropbox built a state-of-the-art Optical Character Recognition (OCR) pipeline.  

This post will be updated multiple times as we will deal with a lot of experiments spanning multiple weeks. 

> All the codes safely stored in github repo [Keras]()

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.

<p align="center">
<img src='/images/dl_project/deep_learning_project.jpg' width="50%"/> 
</p>



Feel free to jump anywhere,

- [Steps in DL Project](#steps-in-dl-project)
- [Dropbox Case Study](#dropbox-case-study)
- [Our Application](#our-application)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)


Full Stack Deep Learning Bootcamp provides an excellent guide on many different questions keeping up DL practitioner such as, "How to " As the graphics below describes, this course was really about practices in creating production-ready projects.

<p align="center">
<img src='/images/dl_project/course.png' width="50%"/> 
</p>




We will steal some of the slides from the lectures that serve as an excellent to-do's as a starting point of any DL project. We will apply all these ideas and answer all the questions in context of our project.

### Steps in DL Project

We start with outlining these steps that are performed for every project in Deep Learning.

<p align="center">
<img src='/images/dl_project/steps.png' width="50%"/> 
</p>

1. Planning & Project setup 

In this first step, we lay out what are the goals of the project. We determine what the requirement the project needs and make sure have enough resources allocated for the project.

2. Data collection & labelling

Here comes the most critical step, we look for different ways in which we can collect appropriate data for our project and look for cheap ways to label the data. Here we focus on the questions, How hard is to get data?, How expensive is data labelling? or How much data will be needed?

3. Training & Debugging

In this step, we start with implementing baselines. We look for any SoTA models and reproduce those. All we need is to make our model more robust and effective. We also look at what metric do we care about and decide which metric to the model optimize for.

4. Deploying & Testing

In last step, we write test functions to test out the model in some version control (not after deploying in real-world but before) to check the robustness of the model and once happy with the results we are ready to deploy.

Notice the flow is not linear or sequential, there is a lot of backtracking and improving as we improve our beliefs about the project. One example could be that having decided goal and collected data, we moved on to training step but once there we realize that our labels are unreliable or realize that goal is too hard and thus we backtrack to second or first step from third step. We keep updating everything as new information keeps poping everytime

<p align="center">
<img src='/images/dl_project/more_steps.png' width="50%"/> 
</p>


### Dropbox Case Study

Here I will take you through the journey of how the team at Dropbox built and deployed a state-of-the-art OCR pipeline to millions of users.

- [x] Planning & Project setup

The goal of the project was to enable following features for Dropbox Business users
- Extract all the text in scanned documents and index it, so that it can be searched for later
- Create a hidden overlay so text can be copied and pasted from the scans saved as PDFs

The first version used a commerical off-the-shelf OCR Library before creating own machine-learning OCR system. Once they confirmed that there was indeed strong user demand for the mobile document scanner and OCR, they decided to build our own in-house OCR system.

Another aspect of encourage the project was cost consideration. Having own OCR system would save them significant money as the licensed commercial OCR SDK charged them based on the number of scans.

- [x] Data collection & labelling

To collect data, they asked a small percentage of users whether they would donate some of their image files to improve OCR algorithms. The most important factor taken into consideration at Dropbox was privacy. The donated files were kept private and secure by not keeping donated data on local machines in permanent storage, maintaining extensive auditing, requiring strong authentication to access any of it, and more.

Next step was how to label this user-donated data. One way is to use platform such as [Amazon’s Mechanical Turk](https://www.mturk.com/mturk/welcome) (MTurk) but the dataset would be exposed in the wild to workers. To navigate this challenge, the team created their own platform for data annotation, named DropTurk. They hired contractors under a strict non-disclosure agreement (NDA) to ensure that they cannot keep or share any of the data they label. Here is an example of DropTurk UI for adding ground truth for word images.

<p align="center">
<img src='/images/dl_project/dropturk.png' width="50%"/> 
</p>

Using this platform, the team collected both word-level dataset, which has images of individual words and their annotated text, as well as a full document-level dataset, which has images of full documents (like receipts) and fully transcribed text.

- [x] Training & Debugging

Start with simple network and simple version of the goal. The team at Dropbox started with simple goal to turning an image of a single word into text. To train this network, they needed data. Back to previous step, they decided to use synthetic data. To gather synthetic data, they created a pipeline of 3 pieces, first a corpus of words to use, second a collection of fonts for drawing the words and third a set of geometric and photometric transformations meant to simulate real world distortions. 

Here is a sample of synthetic dataset for generating word images,

<p align="center">
<img src='/images/dl_project/synthetic_sample.png' width="50%"/> 
</p>

The team started with with words coming from a collection of [Project Gutenberg](https://www.gutenberg.org/) books from the 19th century, about a thousand fonts they collected, and some simple distortions like rotations, underlines, and blurs. They generated about a million synthetic words, trained a deep net, and then tested the accuracy, which was around 79%. 

Here are some highlights which the team provides to improve the recognition accuracy. 

- The network didn't perform well on receipts, so word corpus was expanded to add [Uniform Product Code](https://en.wikipedia.org/wiki/Universal_Product_Code) (UPC) database.
- The network was struggling with letters with disconnected segments. Receipts are often printed with thermal fonts that have stippled, disconnected, or ink smudged letters, but the network had only been given training data with smooth continuous fonts (like from a laser printer) or lightly bit-mapped characters. To overcome this, they included words on thermal printer fonts to get same effect as the characters appear on any receipt.
- The team did research on top 50 fonts in the world and created a font frequency system that allowed them to sample from common fonts (such as Helvetica or Times New Roman) more frequently, while still retaining a long tail of rare fonts (such as some ornate logo fonts). They discovered that some fonts have incorrect symbols or limited support, resulting in just squares, or their lower or upper case letters are mismatched and thus incorrect. By manually going through all 2000 fonts, they marked the fonts that had invalid symbols.
- From a histogram of the synthetically generated words, the team discovered that many symbols were underrepresented, such as / or &. They artifically boosted the frequency of these in the synthetic corpus, by synthetically generating representative dates, prices, URLs, etc.
- They added a large number of visual transformations, such as warping, fake shadows, and fake creases, and much more.




### Our Application

Here is image that gives an overview of what needs to be done for our application.

<p align="center">
<img src='/images/dl_project/full_project.png' width="50%"/> 
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

