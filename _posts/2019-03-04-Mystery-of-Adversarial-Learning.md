---
layout:     post
title:      Mystery of Adversarial Learning
date:       2019-03-04 12:00:00
summary:    This post will provide a brief introduction to 
categories: adversarial learning
published : false
---


# Mystery of Adversarial Learning

In this notebook, 

> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Adversarial%20Learning/adv_learning_keras.ipynb), [PyTorch](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Adversarial%20Learning/adv_learning_pytorch.ipynb) and [Tensorflow](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Adversarial%20Learning/adv_learning_tf.ipynb).

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.

<p align="center">
<img src='/images/adv_learning/meme.jpeg' width="50%"/> 
</p>


Feel free to jump anywhere,

- [Adversarial Learning](#adversarial-learning)
  - [Clever Hans](#clever-hans)
  - [Adversarial Attacks](#adversarial-attacks)
    - [Non-targeted adversarial attack](#non-targeted-adversarial-attack)
    - [Targeted adversarial attack](#targeted-adversarial-attack)
    - [Model stealing techniques](#model-stealing-techniques)
  - [Real World Examples](#real-world-examples)
  - [Adversarial Training](#adversarial-training)
  - [Beyond Images](#beyond-images)
  - [Conclusion](#conclusion)  
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)


# Adverserial Training

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' /> 
</p>


<span class='red'>I-know-everything:</span> My dear young padwan, you are learning quite a lot and honestly, keeping up with the pace of reasearch is really a challenge in itself. But we have tried to cover and focus on the basic concepts in Deep Learning and its application. The journey so far has been like visting a lot of cool stations. We visited MLP, CNN, RNN, LSTM & GRU, and also we visited one the very famous places Transfer Learning in CNN and Transfer Learning in NLP. Hope you enjoyed the journey. A lot is still waiting to be explored. One such topic of interest today is about Adveserial Training. But before that let me tell you a story about clever horse named Clever Hans who could do arithmetic.

<span class='green'>I-know-nothing:</span> Horse doing arithmetic? For real?  

<span class='red'>I-know-everything:</span> Here is the picture of clever hans (a horse) with its owner lived in 1900s. The story of Clever Hans goes like this: 

<p align="center">
<img src='/images/adv_learning/cleverhans.jpg' width="50%"/> 
</p>

### Clever Hans

There was a horse named Clever Hans who could do arithmetic (*Yes you read it right*). The trainer of horse was not any kind of charlatan, didn't want to make any money of any sort but gained a lot of attention but also believed it to be true. You could ask the horse to do *2+2* and then the horse would tap his foot 4 times. Huge crowd of people would gather and watch the horse perform. As far as anybody could tell, it really was able to actually answer wide variety of questions people ask the horse. Later a pyschologist decided to examine the horse. He went to an enclosed area with no other people and wore a mask and he asked the horse to do arithmetic and the horse couldn't do it at all. It turned out what was happening that the horse had not learned arithmetic but *horse had learned how to read people's emotional reactions*. So, you would ask the horse add 1 plus 2 and horse would tap his hoof once and crowd would stare at him in expectation. Then the horse would tap his hoof the second time and everybody in the crowd would sit on the edge and see, then he would tap his hoof the third time and everybody would be like oh my good he knows arithmetic. And then he would stop taping.

So, clever hans was trained to answer these questions. And he found a way of doing it that made him appear to be successful by metric of "Can he provide the right answer where roomful of people are watching him?". He hadn't learned arithmetic and so could not generalize to unusual situations when there weren't a room of people to provide the reactions that he needed to solve the problem. 

<span class='green'>I-know-nothing:</span> Ahh now I see, the horse was clever indeed in reading people's emotional reactions. But how does this relate to Machine Learning I wonder?

<span class='red'>I-know-everything:</span> Great! So, we have seen how a vision model, where we use CNN, was able to attain human level accuracy in classifying images and we also saw what the model is actually looking at when it is classifying the image. If we given the following image of "panda" to the classifer it correctly predicts that image as "panda" but if we add a little(*calculated and not random*) noise to the same image, as you can see the resultant image(*original image + noise*) isn't much different from the original panda image. If we pass this resultant image to the classifier, it predicts the image as "gibbon" with 99% confidence.(😞) This resultant image is called "Adversarial Example". This example is *fooling CNN into thinking that panda is a gibbon.*

<p align="center">
<img src='/images/adv_learning/panda.jpg' width="50%"/> 
</p>

An Adversarial Example is an example that has been carefully computed to be misclassified. To make a new image indistinguishable to human obeserver from original image.
 
<span class='green'>I-know-nothing:</span> So what is really going on? Did the classifier cheat with us the same way Clever hans did? Are there any other methods which we can cheat? Is there any way to defend this cheating? Is it only in images or also in other tasks such as NLP and RL? This cheating can really put the state of the art classifier in a very difficult position as to are they really state of the art(SOTA) in classification and if someone misuses these techniques in fooling the classifier. This certainly has some serious after effects.

<span class='red'>I-know-everything:</span> That is certainly true. This issue of adversarial example does put the mark of SOTA  on classifier really in a jeopardy!

There are mainly 3 types of adversarial attacks. We will explain why is it so easy to perform them, and discuss the security implications that stem from these attacks.

1. Non-targeted adversarial attack
2. Targeted adversarial attack
3. Model stealing techniques

## Adversarial Attacks

We study each attack in-detail here.

A trained CNN model acts as a linear seperator for high dimensional data for different classes where every point(image) is associated with its class. Of course, the boundary of seperation is not perfect. This provides an opportunity to push one image from one class to another (*cross the boundary*) i.e. perturbating the input data in the direction of another class.

<p align="center">
<img src='/images/adv_learning/boundary.png' width="50%"/> 
</p>

A better way to illustrate the two non-targeted and targeted attack is explained by this story of Sherlock Holmes on [cleverhans](http://www.cleverhans.io/security/privacy/ml/2016/12/16/breaking-things-is-easy.html) blog :

> Suppose Professor Moriarty wishes to frame Sherlock Holmes for a crime. He may arrange for an unsuspected accomplice to give Sherlock Holmes a pair of very unique and ornate boots. After Sherlock has worn these boots in the presence of the policemen he routinely assists, the policemen will learn to associate the unique boots with him. Professor Moriarty may then commit a crime while wearing a second copy of the same pair of boots, leaving behind tracks that will cause Holmes to fall under suspicion.

In machine learning, the strategy followed by the adversary is to perturb training points in a way that increases the prediction error of the machine learning when it is used in production. The simplest yet still very efficient algorithm is known as Fast Gradient Step Method (FGSM) is used by both the attacks to generate adversarial examples(*very fast*) introduced in [this](https://arxiv.org/pdf/1412.6572.pdf) paper by Goodfellow and colleagues at Google. The core idea is to add some defined $$\epsilon$$ weak noise on every step of optimization, drifting towards the desired class (targeted) — or, if you wish, away from the correct one (non-targeted).

$$
\begin{aligned}
x^{adv} & = x + \epsilon * sign(\nabla_{x} J(x, y_{true})) \\
\textbf{where}, x &= \textbf{clean image} \\
x^{adv} &= \textbf{perturbed adversarial image} \\
J &= \textbf{classification loss} \\
y_{true} &= \textbf{true label for input image x} \\
\end{aligned}
$$


### Non-targeted adversarial attack

Non-targeted adversarial attack uses FGSM to makes the classifier to give incorrect result of any other class than input image class. Here the objective is to perturb the input image in direction where the gradient increases error by some $$\epsilon$$ in such a way that when we reconstruct the resultant adversarial image it looks indistinguishable than the original image.

```python
def non_targeted_attack(img):
    img = img.cuda()
    label = torch.zeros(1, 1).cuda()

    x, y = Variable(img, requires_grad=True), Variable(label)
    for step in range(steps):
        zero_gradients(x)
        out = model(x)
        y.data = out.data.max(1)[1]
        _loss = loss(out, y)
        _loss.backward()
        normed_grad = step_alpha * torch.sign(x.grad.data)
        step_adv = x.data + normed_grad
        adv = step_adv - img
        adv = torch.clamp(adv, -eps, eps)
        result = img + adv
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result
    return result.cpu(), adv.cpu()
```

<p align="center">
<img src='/images/adv_learning/non_targeted.png' width="50%"/> 
</p>


### Targeted adversarial attack

Targeted adversarial attack uses FGSM to makes the classifier to give incorrect result of specific class for given input image. The main change is the sign of the gradient. As opposed to the non-targeted attack, where the goal was to increase the error assuming that the targeted model is almost always correct, here we are going to minimize the error. Here we minimize the error by computing loss with respect to given (incorrect target) label such that when attack completes, the image outputs that it belongs to the specific class making the attack successful.

```python
def targeted_attack(img, label_idx):
    img = img.cuda()
    label = torch.Tensor([label_idx]).long().cuda()

    x, y = Variable(img, requires_grad=True), Variable(label)
    for step in range(steps):
        zero_gradients(x)
        out = model(x)
        _loss = loss(out, y)
        _loss.backward()
        normed_grad = step_alpha * torch.sign(x.grad.data)
        step_adv = x.data - normed_grad
        adv = step_adv - img
        adv = torch.clamp(adv, -eps, eps)
        result = img + adv
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result
    return result.cpu(), adv.cpu()
```

<p align="center">
<img src='/images/adv_learning/targeted.png' width="50%"/> 
</p>

Here is one example by Goodfellow et al in using [2d Adversarial Objects in fooling neural networks](https://bengio.abracadoudou.com/publications/pdf/kurakin_2017_iclr_physical.pdf),

<p align="center">
<img src='/images/adv_learning/washing_machine.png' width="50%"/> 
</p>

Here is one example from [lab six](https://www.labsix.org/) where they use [3d Adversarial Objects in fooling neural networks](https://www.labsix.org/physical-objects-that-fool-neural-nets/),

<p align="center">
 <video width="320" height="240" controls>
  <source src="/images/adv_learning/turtle.mp4" type="video/mp4">
</video> 
</p>

### Model stealing techniques

Model stealing Techniques are used to “steal” (i.e., duplicate) models or recover training data membership via blackbox probing.
Both the above attacks can be considered as whitebox attacks where the attacker has access to the model’s parameters (gradient in this case) whereas in black box attacks, the attacker has no access to these parameters, i.e., it uses a different model or no model at all to generate adversarial images with the hope that these will transfer to the target model. In the black-box settings, the machine learning model is said to act as an *oracle*. A strategy is to first query the oracle in order to extract an approximation of its decision boundaries—the substitute model—and then use that extracted model to craft adversarial examples that are misclassified by the oracle. This is one of the attacks that exploit the transferability of adversarial examples: they are often misclassified simultaneously across different models solving the same machine learning task, despite the fact that these models differ in their architecture or training data.

Here is one example from [lab six](https://www.labsix.org/) where they use [Partial Information Attacks on Real-world AI](https://www.labsix.org/partial-information-adversarial-examples/),

<p align="center">
 <video width="320" height="240" controls>
  <source src="/images/adv_learning/black_box.mp4" type="video/mp4">
</video> 
</p>

## Real World Examples

1. Print a “noisy” ATM check written for $100 – and cash it for $1,000,000.
2. [Swap](https://arxiv.org/pdf/1511.07528.pdf) a road sign with a slightly perturbed one that would set the speed limit to 200 – in a world of self-driving cars it can be quite dangerous.
3. Don’t wait for self-driving cars – redraw your license plate and cameras will never recognize your car.
4. [Cause](http://openaccess.thecvf.com/content_ECCV_2018/papers/Arjun_Nitin_Bhagoji_Practical_Black-box_Attacks_ECCV_2018_paper.pdf) an NSFW detector to incorrectly recognize an image as safe-for-work
5. [Cause](https://arxiv.org/pdf/1811.03194.pdf) an ad-blocker to incorrectly identify an advertisement as natural content
6. [Cause](https://nicholas.carlini.com/papers/2016_usenix_hiddenvoicecommands.pdf) a digital assistant to incorrectly recognize commands it is given
7. [Cause](https://www.covert.io/research-papers/deep-learning-security/Large-scale%20Malware%20Classification%20using%20Random%20Projections%20and%20Neural%20Networks.pdf) a malware (or spam) classifier to identify a malicious file (or spam email) as benign 

<p align="center">
<img src='/images/adv_learning/traffic_sign.png' width="50%"/> 
</p>

And imagination is limit. There is so many bad examples which can be exploited. Just like any new technology not designed with security in mind, when deploying a machine learning system in the real-world, there will be adversaries who wish to cause harm as long as there exist incentives(i.e., they benefit from the system misbehaving).

## Adversarial Training

What can be done? How can we avoid Adversarial attacks? From above examples we can infer that Adversarial Examples are security concern. Thus there is need to create a robust machine learning algorithm such that if a powerful adversary who is intentionally trying to cause a system to misbehave cannot succeed.



One way for Adversarial Training is to proactively generate adversarial examples as part of the training procedure. We have already seen how we can leverage FGSM to generate adversarial examples inexpensively in large batches. The model is then trained to assign the same label to the adversarial example as to the original example—for example, we might take a picture of a cat, and adversarially perturb it to fool the model into thinking it is a vulture, then tell the model it should learn that this picture is still a cat. 






## Beyond Images

Adversarial examples are not limited to image classification. Adversarial examples are seen in [speech recognition](https://arxiv.org/pdf/1801.01944), [question answering systems](https://arxiv.org/pdf/1707.07328), [reinforcement learning](https://arxiv.org/abs/1702.02284), and other tasks.

[Here]() is video demonstrating adversarial example in speech recognition.

<p align="center">
<img src='/images/adv_learning/text_adv.png' width="50%"/> 
</p>

[Here](https://www.youtube.com/watch?&v=r2jm0nRJZdI) is video demonstrating adversarial example in RL.


## Conclusion


The study of adversarial examples is exciting because many of the most important problems remain open, both in terms of theory and in terms of applications. 
On the theoretical side, no one yet knows whether defending against adversarial examples is a theoretically hopeless endeavor (like trying to find a universal machine learning algorithm) or if an optimal strategy would give the defender the upper ground (like in cryptography and differential privacy). 
On the applied side, no one has yet designed a truly powerful defense algorithm that can resist a wide variety of adversarial example attack algorithms.


Well that really concludes adversarial machine learning. Where to next? <span class='purple'>Power of GAN</span>. 

<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology

FGSM - Fast Gradient Sign Method

CNN - Convolution Neural Networks

RL - Reinforcement Learning

GAN - Generative Adversarial Networks

---

# Further Reading

Stanford CS231n 2017 [Lecture 16 | Adversarial Examples and Adversarial Training](https://www.youtube.com/watch?v=CIfsB_EYsVI)

Nicholas Carlini's [Adversarial Machine Learning Reading List](https://nicholas.carlini.com/writing/2018/adversarial-machine-learning-reading-list.html)

[Explaining and Harnessing Adversarial Examples](https://arxiv.org/pdf/1412.6572.pdf)

[Adversarial Examples Are Not Easily Detected:Bypassing Ten Detection Methods](https://nicholas.carlini.com/papers/2017_aisec_breakingdetection.pdf)

[Adversarial Examples in Real Physical World](https://bengio.abracadoudou.com/publications/pdf/kurakin_2017_iclr_physical.pdf)

[On Evaluating Adversarial Robustness](https://arxiv.org/pdf/1902.06705.pdf)

[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/pdf/1706.06083.pdf)

[Towards Evaluating the Robustnessof Neural Networks](https://arxiv.org/pdf/1608.04644.pdf)

[Synthesizing Robust Adversarial Examples](https://arxiv.org/pdf/1707.07397)

[Black-box Adversarial Attacks with Limited Queries and Information](https://arxiv.org/pdf/1804.08598)

[Wild Patterns: Ten Years After the Rise ofAdversarial Machine Learning](https://arxiv.org/pdf/1712.03141.pdf)

cleverhans blog: [Breaking things is easy](http://www.cleverhans.io/security/privacy/ml/2016/12/16/breaking-things-is-easy.html), [Is attacking machine learning easier than defending it?](www.cleverhans.io/security/privacy/ml/2017/02/15/why-attacking-machine-learning-is-easier-than-defending-it.html) and [The challenge of verification and testing of machine learning](http://www.cleverhans.io/security/privacy/ml/2017/06/14/verification.html)

[How Adversarial Attacks Work](https://blog.ycombinator.com/how-adversarial-attacks-work/)

Gradient Science's blog: [A Brief Introduction to Adversarial Examples](http://gradientscience.org/intro_adversarial/), [Training Robust Classifiers (Part 1)](http://gradientscience.org/robust_opt_pt1/) and [Training Robust Classifiers (Part 2)](http://gradientscience.org/robust_opt_pt2/)

Elie's blog on [Attacks against machine learning — an overview](https://elie.net/blog/ai/attacks-against-machine-learning-an-overview/)



---

# Footnotes and Credits

[Meme](https://medium.com/@ml.at.berkeley/tricking-neural-networks-create-your-own-adversarial-examples-a61eb7620fd8)

[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)

[Turtle Video](https://www.labsix.org/physical-objects-that-fool-neural-nets/)

[Traffic sign](https://www.cc.gatech.edu/news/611783/erasing-stop-signs-shapeshifter-shows-self-driving-cars-can-still-be-manipulated)

---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

