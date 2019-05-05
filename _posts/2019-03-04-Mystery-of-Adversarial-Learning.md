---
layout:     post
title:      Mystery of Adversarial Learning
date:       2019-03-04 12:00:00
summary:    This post will provide a brief introduction to adversarial machine learning where we introduce to different attacks and defense methods and give the examples in real-world scenarios.
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
    - [Gradient-based adversarial attack](#gradient-based-adversarial-attack)
    - [Optimization-based adversarial attack](#optimization-based-adversarial-attack)
    - [Model stealing techniques](#model-stealing-techniques)
  - [Real World Examples](#real-world-examples)
  - [Defenses against Adversarial Attacks](#defenses-against-adversarial-attacks)
  - [Evaluating Adversarial Robustness](#evaluating-adversarial-robustness)
  - [Beyond Images](#beyond-images)
  - [Conclusion](#conclusion)  
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)


# Adversarial Training

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' /> 
</p>


<span class='red'>I-know-everything:</span> My dear young padwan, you are learning quite a lot and honestly, keeping up with the pace of on-going reasearch in the field of deep learning, it's is really a challenge in itself. But we have tried to cover and focus on the basic concepts in Deep Learning and its application. The journey so far has been like visting a lot of cool stations. We visited [MLP](https://dudeperf3ct.github.io/mlp/mnist/2018/10/08/Force-of-Multi-Layer-Perceptron/), [CNN](https://dudeperf3ct.github.io/cnn/mnist/2018/10/17/Force-of-Convolutional-Neural-Networks/), [RNN](https://dudeperf3ct.github.io/rnn/2019/01/19/Force-of-Recurrent-Neural-Networks/), [LSTM & GRU](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/), and also we visited one the very famous places [Transfer Learning in CNN](https://dudeperf3ct.github.io/transfer/learning/catsvsdogs/2018/11/20/Power-of-Transfer-Learning/) and [Transfer Learning in NLP](https://dudeperf3ct.github.io/nlp/transfer/learning/2019/02/22/Power-of-Transfer-Learning-in-NLP/). Hope you enjoyed the journey. A lot is still waiting to be explored. One such topic of interest today is about *Adversarial Learning*. But before that let me tell you a story about *clever horse named Clever Hans who could do arithmetic*.

<span class='green'>I-know-nothing:</span> Horse doing arithmetic? For real?  

<span class='red'>I-know-everything:</span> Here is the picture of clever hans (a horse) with its owner lived in 1900s.

<p align="center">
<img src='/images/adv_learning/cleverhans.jpg' width="50%"/> 
</p>

### Clever Hans

There was a horse named Clever Hans who could do arithmetic (*Yes you read it right*). The trainer of horse was not any kind of charlatan, didn't want to make any money of any sort but gained a lot of attention and also believed it to be true. You could ask the horse to do *2+2* and then the horse would tap his foot 4 times. Huge crowd of people would gather and watch the horse perform. As far as anybody could tell, it really was able to actually answer wide variety of arithmetic questions people ask the horse. Later a psychologist decided to examine the horse. He went to an enclosed area with no other people and wore a mask and he asked the horse to do arithmetic and the horse couldn't do it at all. <span class='red'>It turned out what was happening that the horse had not learned arithmetic but *horse had learned how to read people's emotional reactions*.</span> So, you would ask the horse add 1 plus 2 and horse would tap his hoof once and crowd would stare at him in expectation. Then the horse would tap his hoof the second time and everybody in the crowd would sit on the edge and see, then he would tap his hoof the third time and everybody would be like oh my good he knows arithmetic. And then he would stop taping.

So, clever hans was trained to answer these questions. And he found a way of doing it that made him appear to be successful by metric of "Can he provide the right answer where roomful of people are watching him?". He hadn't learned arithmetic and so could not generalise to unusual situations when there weren't a room of people to provide the reactions that he needed to solve the problem. 

<span class='green'>I-know-nothing:</span> Ahh now I see, the horse indeed was clever in reading people's emotional reactions. But how does this relate to Machine Learning I wonder?

<span class='red'>I-know-everything:</span> Great! So, we have seen how a vision model, where we use CNN, was able to attain human level accuracy in classifying images and we also saw [what the model is actually looking](https://dudeperf3ct.github.io/visualize/cnn/catsvsdogs/2018/12/02/Power-of-Visualizing-Convolution-Neural-Networks/) at when it is classifying the image. If we give the following image of "panda" to the classifier it correctly predicts that image as "panda" but if we add a little(*calculated and not random*) noise to the same image, as you can see the resultant image(*original image + noise*) isn't much different from the original panda image. If we pass this resultant image to the classifier, it predicts the image as "gibbon" with 99% confidence.(😞) This resultant image is called "Adversarial Example". This example is *fooling CNN into thinking that panda is a gibbon.*

<p align="center">
<img src='/images/adv_learning/panda.jpg' width="50%"/> 
</p>

<span class='green'>An Adversarial Example is an example that has been carefully computed to be misclassified. To make a new image indistinguishable to human obeserver from original image. Adversaries can craftily manipulate legitimate inputs, which may be imperceptible to human eye, but can force a trained model to produce incorrect outputs.</span>

<span class='green'>I-know-nothing:</span> So what is really going on? *Did the classifier cheat with us the same way Clever hans did? Are there any other methods which we can cheat? Is there any way to defend this cheating? Is it only in images or also in other tasks such as NLP and RL?* This cheating can really put the state of the art classifier in a very difficult position as to are they really state of the art(SOTA) in classification and if someone misuses these techniques in fooling the classifier. This certainly has some serious after effects.

<span class='red'>I-know-everything:</span> That is certainly true. This issue of adversarial example does put the mark on SOTA classifier really in a jeopardy! <span class='purple'>Are they really good as they claim, beating humans?</span>

There are mainly **3 types of adversarial attacks**. We will explain why is it so easy to perform them, and discuss the security implications that stem from these attacks.

1. Gradient-based adversarial attack
2. Optimization-based adversarial attack 
3. Model stealing techniques

## Adversarial Attacks

A trained CNN model acts as a linear seperator for high dimensional data for different classes, where every point(image) is associated with its class. Of course, the boundary of seperation is not perfect. <span class='green'>This provides an opportunity to push one image from one class to another (*cross the boundary*) i.e. perturbing the input data in the direction of another class.</span>

<p align="center">
<img src='/images/adv_learning/boundary.png' width="50%"/> 
</p>

A better way to illustrate the two, non-targeted and targeted attack is explained by the story of Sherlock Holmes on [cleverhans](http://www.cleverhans.io/security/privacy/ml/2016/12/16/breaking-things-is-easy.html) blog :

> Suppose Professor Moriarty wishes to frame Sherlock Holmes for a crime. He may arrange for an unsuspected accomplice to give Sherlock Holmes a pair of very unique and ornate boots. After Sherlock has worn these boots in the presence of the policemen he routinely assists, the policemen will learn to associate the unique boots with him. Professor Moriarty may then commit a crime while wearing a second copy of the same pair of boots, leaving behind tracks that will cause Holmes to fall under suspicion.

In machine learning, the strategy followed by the adversary is to perturb training points in a way that increases the prediction error of the machine learning when it is used in production. The simplest yet still very efficient algorithm is known as Fast Gradient Step Method (FGSM) is used by both the attacks to generate adversarial examples(*very fast*) introduced in [this](https://arxiv.org/pdf/1412.6572.pdf) paper by Goodfellow and colleagues at Google. <span class='saddlebrown'>The core idea is to add some defined $$\epsilon$$ weak noise on every step of optimization, drifting towards the desired class (targeted) — or, if you wish, away from the correct one (non-targeted).</span>

$$
\begin{aligned}
x^{adv} & = x + \epsilon * sign(\nabla_{x} J(x, y_{true})) \\
\textbf{where}, x &= \textbf{clean image} \\
x^{adv} &= \textbf{perturbed adversarial image} \\
J &= \textbf{classification loss} \\
y_{true} &= \textbf{true label for input image x} \\
\end{aligned}
$$

### Gradient-based adversarial attack

These are the simplest technique that demonstrate the linearity of neural networks using Fast-Gradient Sign Method(FGSM) and as the name suggest these are gradient-based methods.

#### Non-targeted adversarial attack

<span class='saddlebrown'>Non-targeted adversarial attack uses FGSM to makes the classifier to give incorrect result of any other class than input image class.</span> Here the objective is to perturb the input image in direction where the gradient increases error by some $$\epsilon$$ in such a way that when we reconstruct the resultant adversarial image it looks indistinguishable than the original image.

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


#### Targeted adversarial attack

<span class='saddlebrown'>Targeted adversarial attack uses FGSM to makes the classifier to give incorrect result of specific class for given input image.</span> The main change is the sign of the gradient. As opposed to the non-targeted attack, where the goal was to increase the error assuming that the targeted model is almost always correct, here we are going to minimize the error. Here we minimize the error by computing loss with respect to given (incorrect target) label such that when attack completes, the image outputs that it belongs to the specific class making the attack successful.

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
<img src='/images/adv_learning/washing_machine.png' width="80%"/>
</p>

Here is one example from [lab six](https://www.labsix.org/) where they use [3d Adversarial Objects in fooling neural networks](https://www.labsix.org/physical-objects-that-fool-neural-nets/),

<p align="center">
<img src='/images/adv_learning/turtle.gif' width="50%"/> 
</p>

These are not only the gradient-based adversarial attacks but are the simplest. 

### Optimization-based adversarial attack

<span class='saddlebrown'>C&W attack introduced in [Towards Evaluating the Robustnessof Neural Networks](https://arxiv.org/pdf/1608.04644.pdf) is by far one of the strongest attacks.</span> They formulate targeted adversarial attacks as an optimization problem, take advantage of the internal configurations of a targeted DNN for attack guidance, and use the $$L_{2}$$ norm (i.e. Euclidean distance) to quantify the difference between the adversarial and the original examples. In particular, the representation in the logit layer (the layer prior to the final fully connected layer) is used as an indicator of attack effectiveness. Consequently, the C&W attack can be viewed as a gradient-descent based targeted adversarial attack driven by the representation of the logit layer of a targeted DNN and the $$L_{2}$$ distortion. <span class='saddlebrown'>C&W attack picks random multiple random starting points close to the original image and run gradient descent from each of those points for a fixed number of iterations.</span> They tried three optimizers — standard gradient descent, gradient descent with momentum, and Adam — and all three produced identical-quality solutions. However, Adam converges substantially more quickly than the others.


### Model stealing techniques

<span class='saddlebrown'>Model stealing Techniques are used to “steal” (i.e., duplicate) models or recover training data membership via blackbox probing.</span> Both the above attacks can be considered as whitebox attacks where the attacker has access to the model’s parameters (gradient in this case) whereas in black box attacks, the attacker has no access to these parameters, i.e., it uses a different model or no model at all to generate adversarial images with the hope that these will transfer to the target model. 

In the black-box settings, the machine learning model is said to act as an *oracle*. One strategy in using black-box setting for stealing called *oracle attack* is to first query the oracle in order to extract an approximation of its decision boundaries—the substitute model—and then use that extracted model to craft adversarial examples that are misclassified by the oracle. This is one of the attacks that exploit the transferability of adversarial examples: they are often misclassified simultaneously across different models solving the same machine learning task, despite the fact that these models differ in their architecture or training data.

Here is one example from [lab six](https://www.labsix.org/) where they use [Partial Information Attacks on Real-world AI](https://www.labsix.org/partial-information-adversarial-examples/) another black-box attack,

<p align="center">
<img src='/images/adv_learning/black_box.gif' width="50%"/> 
</p>

Sometimes perturbing too many pixels can make the modified image seem perceptible to human eye. Su et al proposed [a method](https://arxiv.org/pdf/1710.08864.pdf) by perturbing only one pixel with differential evolution using black-box setting.  

<p align="center">
<img src='/images/adv_learning/one_pixel.png' width="50%"/> 
</p>

Changing one pixel turns ship into 99.7% car, horse into 99.9% frog or a deer into airplane. This means we cannot just randomly select any pixel from image, it has to be specific for it to work. This is where Differential Evolution comes into play. DE belongs to the general class of evolutionary algorithms which does not use the gradient information for optimizing and therefore  do not require the objective function to be differentiable. As with typical EA algorithms during each iteration, set of candidate solutions is generated according to current population. Then children are compared with their corresponding parent surviving if they are more fitted than their parents. The last surviving child is used to alter the pixel in the image. And this is how from random pixels DE chooses one pixel which confidently changes the class to input image. 

## Real World Examples

1. Print a “noisy” ATM check written for $100 – and cash it for $1,000,000.
2. [Swap](https://arxiv.org/pdf/1511.07528.pdf) a road sign with a slightly perturbed one that would set the speed limit to 200 – in a world of self-driving cars it can be quite dangerous.
3. Don’t wait for self-driving cars – redraw your license plate and cameras will never recognise your car.
4. [Cause](http://openaccess.thecvf.com/content_ECCV_2018/papers/Arjun_Nitin_Bhagoji_Practical_Black-box_Attacks_ECCV_2018_paper.pdf) an NSFW detector to incorrectly recognise an image as safe-for-work
5. [Cause](https://arxiv.org/pdf/1811.03194.pdf) an ad-blocker to incorrectly identify an advertisement as natural content
6. [Cause](https://nicholas.carlini.com/papers/2016_usenix_hiddenvoicecommands.pdf) a digital assistant to incorrectly recognize commands it is given
7. [Cause](https://www.covert.io/research-papers/deep-learning-security/Large-scale%20Malware%20Classification%20using%20Random%20Projections%20and%20Neural%20Networks.pdf) a malware (or spam) classifier to identify a malicious file (or spam email) as benign 

<p align="center">
<img src='/images/adv_learning/traffic_sign.png' width="50%"/> 
</p>

Here is a recent [demo](https://v.qq.com/x/page/x0855xzykn4.html) by Tencent Keen Security Lab which conducted research on Autopilot of Tesla Model S and achieved 3 flaws, *Auto-wipers Vision Recognition Flaw*, *Lane Recognition Flaw* and *Control Steering System with Gamepad*. For more details on the technical details, [here](https://keenlab.tencent.com/en/whitepapers/Experimental_Security_Research_of_Tesla_Autopilot.pdf) is the paper and must watch [video](https://www.youtube.com/watch?v=6QSsKy0I9LE) demonstrating each of the flaws. *Controlling Tesla steering with Gamepad, finally all GTA practise paying off.*

And imagination is limit. There are so many bad examples which can be exploited. <span class='red'>Just like any new technology not designed with security in mind, when deploying a machine learning system in the real-world, there will be adversaries who wish to cause harm as long as there exist incentives(i.e., they benefit from the system misbehaving).</span>

## Defenses against Adversarial Attacks

What can be done? How can we avoid Adversarial attacks? From criticality of above real-world examples we can infer that Adversarial Examples are security concern. Thus there is need to create a robust machine learning algorithm such that if a powerful adversary who is intentionally trying to cause a system to misbehave cannot succeed. *Adversarial training* can defend against FGSM attack by causing gradient masking, where locally the gradient around a given image may point in a direction that is not useful for generating an adversarial example. 

One way for Adversarial training is to proactively generate adversarial examples as part of the training procedure. We have already seen how we can leverage FGSM to generate adversarial examples inexpensively in large batches. The model is then trained to assign the same label to the adversarial example as to the original example—for example, we might take a picture of a cat, and adversarially perturb it to fool the model into thinking it is a vulture, then tell the model it should learn that this picture is still a cat. <span class='saddlebrown'>Adversarial training is a standard brute force approach where the defender simply generates a lot of adversarial examples and augments these perturbed data while training the targeted model.</span> Adversarial training of a model is useful only on adversarial examples which are crafted on the original model. The defense is not robust for black-box attacks where an adversary generates malicious examples on a locally trained substitute model. 

<span class='saddlebrown'>Another way is gradient hiding which consists of hiding information about model's gradient from adversary by using non-differentiable models such as a Decision Tree, a NearestNeighbor Classifier, or a Random Forest.</span> However, this defense are easily fooled by learning a surrogate Black-Box model having gradient and crafting examples using it. The attacker can train their own model, a smooth model that has a gradient, make adversarial examples for their model, and then deploy those adversarial examples against our non-smooth model.

There are many different defenses such as [Defensive Distillation](https://arxiv.org/pdf/1511.04508), image processing methods such as [scalar quantization, spatial smoothing filter](https://arxiv.org/pdf/1705.08378.pdf), [squeezing color bits and local/non-local spatial smoothing](https://arxiv.org/pdf/1704.01155.pdf) and [many more](https://paperswithcode.com/task/adversarial-defense).

## Evaluating Adversarial Robustness

<span class='saddlebrown'>The competition between attacks and defenses for adversarial examples becomes an “arms race”: a defensive method that was  proposed to prevent existing attacks was later shown to be vulnerable to some new attacks, and vice versa. Some defenses showed that they could defend a  particular attack, but later failed with a slight change of the attack. Hence, the evaluation on the robustness of a deep neural network is necessary.</span> Nicholas Carlini et al in [On Evaluating Adversarial Robustness](https://arxiv.org/pdf/1902.06705.pdf) outlines three common reasons why one might be interested in evaluating the robustness of a machine learning model which are, *To defend against an adversary who will attack the system*, *to test the worst-case robustness of machine learning algorithms* and *to measure progress of machine learning algorithms towards human-level abilities*. Adversarial robustness is a measure of progress in machine learning that is orthogonal to performance.


## Beyond Images

Adversarial examples are not limited to image classification. Adversarial examples are seen in [speech recognition](https://arxiv.org/pdf/1801.01944), [question answering systems](https://arxiv.org/pdf/1707.07328), [reinforcement learning](https://arxiv.org/abs/1702.02284), [object detection and semantic segmentation](https://openaccess.thecvf.com/content_iccv_2017/html/Xie_Adversarial_Examples_for_ICCV_2017_paper.html) and other tasks.

**Speech Recognition**

[Here](https://www.youtube.com/watch?v=HvZAZFztlO0) is video demonstrating adversarial example in speech recognition.

**Question Answering Systems**

<p align="center">
<img src='/images/adv_learning/text_adv.png' width="50%"/> 
</p>

**RL**

[Here](https://www.youtube.com/watch?&v=r2jm0nRJZdI) is video demonstrating adversarial example in RL.

**Object Detection and Semantic Segmentation**

<p align="center">
<img src='/images/adv_learning/detection.png' width="50%"/> 
</p>


## Conclusion

- The study of adversarial examples is exciting because many of the most important problems remain open, both in terms of theory and in terms of applications. 
- **On the theoretical side**, <span class='saddlebrown'>no one yet knows whether defending against adversarial examples is a theoretically hopeless endeavour (like trying to find a universal machine learning algorithm) or if an optimal strategy would give the defender the upper ground (like in cryptography and differential privacy).</span> The lacking of proper theoretical tools to describe the solution to these complex optimization problems make it even harder to make any theoretical argument that a particular defense will rule out a set of adversarial examples.
- **On the applied side**, <span class='saddlebrown'>no one has yet designed a truly powerful defense algorithm that can resist a wide variety of adversarial example attack algorithms.</span> Most of the current defense strategies are not adaptive to all types of adversarial attack as one method may block one kind of attack but leaves another vulnerability open to an attacker who knows the underlying defense mechanism.

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

[Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods](https://nicholas.carlini.com/papers/2017_aisec_breakingdetection.pdf)

[Adversarial Examples in Real Physical World](https://bengio.abracadoudou.com/publications/pdf/kurakin_2017_iclr_physical.pdf)

[Adversarial Examples: Attacks and Defenses for Deep Learning](https://arxiv.org/pdf/1712.07107.pdf)

[Adversarial Examples for Evaluating Reading Comprehension Systems](https://arxiv.org/pdf/1707.07328.pdf)

[On Evaluating Adversarial Robustness](https://arxiv.org/pdf/1902.06705.pdf)

[Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/pdf/1608.04644.pdf)

[Synthesizing Robust Adversarial Examples](https://arxiv.org/pdf/1707.07397)

[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/pdf/1706.06083.pdf)

[The limitations of adversarial training and the blind-spot attack](https://openreview.net/pdf?id=HylTBhA5tQ)

[The Limitations of Deep Learningin Adversarial Settings](https://arxiv.org/pdf/1511.07528.pdf)

[One Pixel Attack for Fooling Deep Neural Networks](https://arxiv.org/pdf/1710.08864.pdf)

[Black-box Adversarial Attacks with Limited Queries and Information](https://arxiv.org/pdf/1804.08598)

[Wild Patterns: Ten Years After the Rise of Adversarial Machine Learning](https://arxiv.org/pdf/1712.03141.pdf)

[CAMOU: Learning a Vehicle camouflage for physical adversarial attack on object detectors in the wild](https://openreview.net/pdf?id=SJgEl3A5tm)

[Practical Black-Box Attacks against Machine Learning](https://arxiv.org/pdf/1602.02697.pdf)

[Are Adversarial Examples Inevitable?](https://openreview.net/pdf?id=r1lWUoA9FQ)

[Towards the first adversarially robust neural network model on MNIST](https://openreview.net/pdf?id=S1EHOsC9tX)

[Adversarial Attacks and Defences: A Survey](https://arxiv.org/pdf/1810.00069.pdf)

[Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey](https://arxiv.org/pdf/1801.00553.pdf)

[Adversarial Attacks on Deep Learning Models in Natural Language Processing: A Survey](https://arxiv.org/pdf/1901.06796.pdf)

cleverhans blog: [Breaking things is easy](http://www.cleverhans.io/security/privacy/ml/2016/12/16/breaking-things-is-easy.html), [Is attacking machine learning easier than defending it?](www.cleverhans.io/security/privacy/ml/2017/02/15/why-attacking-machine-learning-is-easier-than-defending-it.html) and [The challenge of verification and testing of machine learning](http://www.cleverhans.io/security/privacy/ml/2017/06/14/verification.html)

[How Adversarial Attacks Work](https://blog.ycombinator.com/how-adversarial-attacks-work/)

Gradient Science's blog: [A Brief Introduction to Adversarial Examples](http://gradientscience.org/intro_adversarial/), [Training Robust Classifiers (Part 1)](http://gradientscience.org/robust_opt_pt1/) and [Training Robust Classifiers (Part 2)](http://gradientscience.org/robust_opt_pt2/)

Elie's blog on [Attacks against machine learning — an overview](https://elie.net/blog/ai/attacks-against-machine-learning-an-overview/)

[Safety and Trustworthiness of Deep Neural Networks: A Survey](https://arxiv.org/pdf/1812.08342v1.pdf)

[Adversarial learning literature](https://github.com/vikramnitin9/adversarial-learning-literature)

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
