---
layout:     post
title:      Power of GAN
date:       2019-04-13 12:00:00
summary:    This post will provide a brief introduction of GANs.
categories: gan
published : false
---


# Generative Adversarial Networks

In this notebook, 

> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/RNN/char_rnn_keras.ipynb), [PyTorch](https://github.com/dudeperf3ct/DL_notebooks/blob/master/RNN/char_rnn_pytorch.ipynb) and [Fastai](https://github.com/dudeperf3ct/DL_notebooks/blob/master/RNN/char_rnn_fastai.ipynb).

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.

<p align="center">
<img src='/images/rnn/rnn_meme.jpg' width="50%"/> 
</p>


Feel free to jump anywhere,

- [Introduction to GAN](#introduction-to-gan)
  - [GAN Framework](#gam-framework)
  - [Cost Functions](#cost-functions)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)



# Introduction to GAN

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' /> 
</p>


<span class='red'>I-know-everything:</span> In our last post, we saw how we can use Adversarial Machine Learning in context of security. We discussed how adversaries can abuse the model and produce malicious results in real world. This name "Adversarial" has different meaning depending on the context. The previous post used Adversarial Training where neural network is used to correctly classify adversarial examples by training the network on adversarial examples. In context of RL, "self play" can be seen as Adversarial Training where the network learns to play with itself. In our today's topic, which is GAN i.e. Generative Adversarial Networks, we will use Adversarial Training where a model is trained on the inputs produced by adversary. Now all the names are cleared, let's get back to the revolutionary GANs. As all posts with GANs start with the [quote](https://www.quora.com/What-are-some-recent-and-potentially-upcoming-breakthroughs-in-deep-learning) from [Yann LeCunn](http://twitter.com/ylecun/), 

> <span class='purple'>"The most important one, in my opinion, is adversarial training (also called GAN for Generative Adversarial Networks). This is an idea that was originally proposed by Ian Goodfellow when he was a student with Yoshua Bengio at the University of Montreal (he since moved to Google Brain and recently to OpenAI).This, and the variations that are now being proposed is the most interesting idea in the last 10 years in ML, in my opinion."</span> #Tradition#not#broken. 

<span class='green'>I-know-nothing:</span> Holding up the tradition, what are GANs? Specifically, what does generative mean?

<span class='red'>I-know-everything:</span> Consider the example where are we are teaching a model to distinguish between dog(y=0) and cat(y=1). The way traditional machine learning classification algorithms like logistic regression or perceptron algorithm tries to find a straight line - a decision boundary - such that when a test image of dog is passed, model checks on which side of decision boundary does the animal falls and makes prediction accordinly. This is what we have learned so far from our past journey, nothing new. But consider a different approach. Instead of learning a decision boundary, what if network learns a model of dog looking at lots of different images of dog and same with cat, a different model of what a cat looks like. Now, when we pass a test image of dog to classify, the test image is matched against the model of dog and model of cat to make prediction. What's difference? Algorithms like logistic regression and perceptron learns mappings directly from space of inputs $$\chi$$ to labels {0, 1} i.e. p(y|x) where y $$\in$$ {0, 1} and these are called <span class='purple'>discriminative learning algorithms.</span> Now, instead of learning the mapping from input space, what if model learns the distribution of input features? This is the idea behind <span class='purple'>generative learning algorithms.</span> They learn p(x|y), for example, if y=0 indicates it's a dog then p(x|y=0) models the distribution of dog's features and p(x|y=1) modelling the distribution of cat's features. GANs belong to the family of generative models. This means that GANs samples data from training set, a distribution of $$p_{data}$$, and learns to represent an estimate of that distribution, resulting in probability distribution $$p_{model}$$. There are cases where model estimates $$p_{model}$$ explicitly and in other cases model is only able to generate samples from $$p_{model}$$. <span class='red'>GAN primarily focuses on the second case generating samples from the model distribution</span> although it is possible to design GANs that can do both.

<p align="center">
<img src='/images/gan/sample_gan.jpg' width="50%"/> 
</p>

Here is an example, where an ideal generative model would be able to train on examples shown on left and then create more examples from the same distribution as shown on the right.

<span class='green'>I-know-nothing:</span> This approach taken by GANs is certainly new compared to our previous approaches of supervised learning. I wonder in what bucket of learning does GAN go in? What's so special about them? What learning function do they use if any?

<span class='red'>I-know-everything:</span> Here's a interesting thing, they belong to both buckets of supervised learning and unsupervised learning. The GAN sets up a supervised learning problem in order to do unsupervised learning. You will understand why so once we introduce different parts of GAN. Let's do that! The basic idea of GAN is setting up game between two players. The two players are <span class='saddlebrown'>generator</span> and <span class='saddlebrown'>discriminator</span>. <span class='purple'>The generator creates samples that are intended to come from the same distribution as the training set. The discriminator examines the samples to determine whether they are real or fake as in, does the input samples belong to the training set or not?</span> So, what is the game between the two players? The generator is trained to fool the discriminator i.e. generator generates a sample and passes it to discriminator. The discriminator using traditional supervised learning is trained to classify the input sample in two classes (real or fake), fooling the discriminator means that discriminator will classify the sample generator to be real instead of fake. And hence, the name "Adversarial". Here we can see that generator wants to be good at fooling discriminator and discriminator wants to be good at classifying samples correctly. This corresponds to [Nash Equilibrium](https://www.youtube.com/watch?v=LJS7Igvk6ZM) from Game Theory. Borrowing example of Alice and Bob from Wikipedia, Alice and Bob are in Nash equilibrium if Alice is making the best decision she can, taking into account Bob's decision while his decision remains unchanged, and Bob is making the best decision he can, taking into account Alice's decision while her decision remains unchanged. Likewise, a group of players are in Nash equilibrium if each one is making the best decision possible, taking into account the decisions of the others in the game as long as the other parties' decisions remain unchanged. GAN requires finding the Nash Equilibrium of the game, which is more difficult than optimizing an objective function as done in traditional machine learning.

Okay, let's try to explain this notion of game through more examples. Suppose the generator to be a counterfeiter, trying to make fake money, and the discriminator as being like police, trying to allow legitimate money and catch counterfeit money. To succeed in this game, the counterfeiter must learn to make money that is indistinguishable from genuine money. Another example, where the generator is person signing cheques using fake signatures and discriminator is bank person, trying to identify if the signature is authentic or not. To succeed in passing the cheque, the person needs to produce signature very real to the original such that bank person gets conned into thinking that fake signature is authentic. You get it right?

Now let's dive in-detail into generator and discriminator.


## GAN Framework

- **Discriminator(D)** is a differentiable function, usually a neural network with parameter $$\theta^{(D)}$$. Discriminative network which takes in input, $$\mathbf{x}$$ "real" which comes from training set or output from generator G($$\mathbf{z}$$) "fake". The goal of discriminator is to classify the input from training set as real and the one from generator as fake. Discriminator is shown half of inputs which are real and remaining half as fakes.

- **Generator(G)** is also differential function, another neural network with parameter $$\theta^{(G)}$$. Generative network takes in input $$\mathbf{z}$$, where $$\mathbf{z}$$ is sample from some prior distribution, G($$\mathbf{z}$$) yields a sample $$\mathbf{x}$$ drawn from $$p_{model}$$. The goal of generator is to fool discriminator.


<p align="center">
<img src='/images/gan/gan.jpg' width="50%"/> 
</p>

Here is the game which is played in two scenarios. In first scenario, left side of the figure, training examples $$\mathbf{x}$$ are randomly sampled from training dataset and used as input for first player, the discriminator(D). The goal of discriminator(D) is to output the probability that its input is real rather than fake. In first scenario, D($$\mathbf{x}$$) tries to be near 1, classifying it to be a real. In second scenario, inputs $$\mathbf{z}$$ to the generator(G) are sampled from model's prior over latent variables. The discriminator then receives the output from generator(G), G($$\mathbf{z}$$), a fake sample generated by generator. Here, the discriminator(D) tries to make D(G($$\mathbf{z}$$)) near 0, as it is fake sample and generator(G) tries to make D(G($$\mathbf{z}$$)) near 1 to fool discriminator in classifying the fake sample as real. If both models have sufficient capacity, then the Nash equilibrium of this game corresponds to the G($$\mathbf{z}$$) being drawn from the same distribution as the training data, and D($$\mathbf{x}$$) = $$\frac{1}{2}$$ for all $$\mathbf{x}$$. How? It will be evident once we introduce cost functions.

## Cost Functions

Above, we mentioned that GAN sets up a supervised learning problem in order to do unsupervised learning. Here is where we will see how that is true. 

### Discriminator's Cost

The discriminative network is a classifier which takes in an input and classifies it to be fake or real ,i.e. 0 or 1. We have seen these types of problems in supervisied learning which go by name binary classifiers. The output of neural network is binary which is constrained by adding sigmoid as last classification layer. As with all supervised algorithms we require objective function to minimize, we also know that there is a particular loss function which corresponds to binary classification, binary cross entropy(BCE). The cost function used for discriminator $$J^{(D)}$$($$\theta^{(D)}$$, $$\theta^{(G)}$$) for parameters $$\theta^{(D)}$$ for discriminative network and $$\theta^{(G)}$$ for generative network is,

We will first define cost function for one data point ($$\mathbf{x}_{1}$$, $$\mathbf{y}_{1}$$) and then generalize over entire dataset for N elements.

$$
\begin{aligned}
J^{(D)}(\theta^{(D)}, \theta^{(G)}) &= -\mathbf{y}_{1}\log_{}D(\mathbf{x}_{1})-(1-\mathbf{y}_{1})(1-D(\mathbf{x}_{1})) \\
&= -\sum_{i=1}^{N}\mathbf{y}_{i}\log_{}D(\mathbf{x}_{i})-\sum_{i=1}^{N}(1-\mathbf{y}_{i})(1-D(\mathbf{x}_{i})) 
\end{aligned}
$$

In GANs, $$x_{i}$$ either come two sources: either $$x_{i}$$ $$\sim$$ $$p_{data}$$, the true distribution, or $$x_{i}$$ = G($$\mathbf{z}$$) where $$\mathbf{z}$$ $$\sim$$ $$p_{model}$$, the generator's distribution, $$\mathbf{z}$$ is sample from some prior distribution. Discriminator sees exactly half of the data coming from each source i.e. half samples are real and remaining half are fake.

$$
\begin{aligned}
J^{(D)}(\theta^{(D)}, \theta^{(G)}) &= -\frac{1}{2} \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}\log_{}D(\mathbf{x}) -\frac{1}{2} \mathbb{E}_{\mathbf{z} \sim p_{z}(\mathbf{z})}\log_{}(1-D(G(\mathbf{z})))
\end{aligned}
$$


## Minmax

To play the game, we need to complete generator's cost function $$J^{(G)}$$. We assume that we are playing the simplest zero-sum game, where the sum of all player's cost is zero. In this zero-sum game, we get $$J^{(D)}$$ + $$J^{(G)}$$ = 0. This gives us $$J^{(G)}$$ = - $$J^{(D)}$$.

From looking at the equations above for $$J^{(D)}(\theta^{(D)}, \theta^{(G)})$$ and figure explaining two scenarios of game, the discriminator decision are accurate when it correctly classifies fake and real samples. In terms of cost function, in first scenario with real samples, D($$\mathbf{x}$$) tries to be near 1, i.e. maximize $$\mathbb{E}_{\mathbf{x} \sim p_{data}}\log_{}D(\mathbf{x})$$. That is, when D($$\mathbf{x}$$) becomes close to 1, $$\mathbb{E}(\log(D(\mathbf{x})))$$ becomes close to 0 and when D($$\mathbf{x}$$) tries to be near 1, $$\mathbb{E}(\log(D(\mathbf{x})))$$ becomes close to $$-\infty$$.  In second scenario with fake samples, D($$\mathbf{x}$$) tries to be near 0, i.e. maximize $$\mathbb{E}_{\mathbf{x} \sim \mathbf{z}}\log_{}(1-D(G(\mathbf{z})))$$.

The generator on other hand is trained to increase the chances of D producing a high probability for a fake example, thus to minimize $$\mathbb{E}_{\mathbf{z}}\log_{}(1-D(G(\mathbf{z})))$$, the part of cost function ($$\mathbb{E}_{\mathbf{x} \sim p_{data}}\log_{}D(\mathbf{x})$$) which deals with real samples will have no effect on generator as it is not sampled from generator.

So, combining both the conclusions from above, <span class='green'>to maximize the cost function for D and minimze the second part of cost function for G, we are essentially playing minmax game.</span>

We substitute V(D, G) = - $$J^{(D)}(\theta^{(D)}, \theta^{(G)})$$ in cost function to get the minmax of value function as follows,

$$
\begin{aligned}
\min_{G} \max_{D} V(D, G) &= \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}[\log_{}D(\mathbf{x})]+ \mathbb{E}_{\mathbf{z} \sim p_{z}(\mathbf{z})}[\log_{}(1-D(G(\mathbf{z})))]
\end{aligned}
$$


<span class='saddlebrown'>The cost used for the generator in the minimax game (equation 10) is useful fortheoretical analysis, but does not perform especially well in practice. In the minimax game, the discriminator minimizes a cross-entropy, but the generator maximizes the same cross-entropy. This is unfortunate for the generator, because when the discriminator successfully rejects generator samples with high confidence, the generator’s gradient vanishes.</span>

To solve this problem, one approach is to continue to use cross-entropy minimization for the generator. Instead of flipping the sign on the discriminator’s cost to obtain a cost for the generator, we flip the target used to construct the cross-entropy cost.  The cost for the generator then becomes:

$$
\begin{aligned}
J^{(G)} &= -\frac{1}{2} \mathbb{E}_{\mathbf{z}}\log_{}(1-D(G(\mathbf{z})))
\end{aligned}
$$

Also, maximum likelihood can be used as cost function for generator,

$$
\begin{aligned}
J^{(G)} &= -\frac{1}{2} \mathbb{E}_{\mathbf{z}}\exp({\sigma^{-1}(D(G(\mathbf{z})))})
\end{aligned}
$$

What is advantage of different cost functions for generators?




## Optimal D

We want to find best or the optimal value for D for fixed G. So, we have cost function, 

$$
\begin{aligned}
\mathbb{E}_{\mathbf{x} \sim p}[f(\mathbf{x})] &= \int p(\mathbf{x})f(\mathbf{x})\,dx  \\   
V(D, G) &= \mathbb{E}_{data}[\log_{}D(\mathbf{x})]+ \mathbb{E}_{generator}[\log_{}(1-D(G(\mathbf{z})))] \\
&= \int_{x} p_{data}(\mathbf{x})\log_{}D(\mathbf{x})  + (\mathbf{x})\log_{}(1-D(G(\mathbf{x})))\,dx
\end{aligned}
$$

To find maximum of above equation, we take derivate and obtain D as,

$$
\begin{aligned}
D(\mathbf{x}) &= \frac{p_{data}}{p_{g}+p_{data}}
\end{aligned}
$$

If G is trained to be optimal i.e. when $$p_{data} \approx p_{g}$$, we obtain optimal D* = $$\frac{1}{2}$$.

## Global Optimal

When both G and D are at optimal values, we have $$p_{data}$$ = $$p_{g}$$ and D* = $$\frac{1}{2}$$, the cost function becomes,

$$
\begin{aligned}
V(D*, G) &= \int_{x} p_{data}(\mathbf{x})\log_{}D(\mathbf{x})  + (\mathbf{x})\log_{}(1-D(G(\mathbf{x})))\,dx\\
&= \log_{}\frac{1}{2}\int_{x}p_{data}\,dx + \log_{}\frac{1}{2}\int_{x}p_{g}\,dx\\
&=-2\log_{}2
\end{aligned}
$$


<span class='green'>I-know-nothing:</span> What is training procedure given that we have two neural networks for D and G? How does backpropogation work? How does G tweak it's parameters based on signal from D?

<span class='red'>I-know-everything:</span> Ahh, excellent questions. The trend in training will be very different than the once observed in standard machine learning algorithms.

## Training GANs

In each step of training, two minibatches are sampled: one minibatch of $$\mathbf{x}$$ values from the dataset for which D will classify as real and another minibatch of $$\mathbf{z}$$ values drawn from the model's prior over latent variables. The two gradient steps are made simultaneously: one updating $$\theta^{(D)}$$ to reduce $$J^{(D)}$$ and one updating $$\theta^{(G)}$$ to reduce $$J^{(G)}$$. In both cases, we use Adam as optimizer.


## Problem in Training GANs

Of course, the training procedure we described above is very unstable. 





## Different types of GANs

GAN literature is filled with different types of GANs or anynameGAN. We will take a peek into some of the GANs.

### WGAN

### DCGAN

### CycleGAN

### StyleGAN

### BigGAN



## Problems in GANs


## Cool Results

Of course, you can't implement all of them. So, let's steal some of the results from all sort of cool GANs.


In next post, we will do something <span class='yellow'>different</span>. We will attempt to dissect any one or two papers. Any suggestions? So, let's call that Paper dissection.

<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology



---

# Further Reading

[NIPS 2016 Tutorial : Generative Adversarial Network](https://arxiv.org/pdf/1701.00160.pdf)

[Generative Learning algorithms](http://cs229.stanford.edu/notes/cs229-notes2.pdf)

[Generative Models by OpenAI](https://blog.openai.com/generative-models/#gan)

[Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)

[WGAN]()

[DCGAN]()

[CycleGAN]()

[StyleGAN]()

[BigGAN]()

[Wasserstein GAN from depthfirstlearning](http://www.depthfirstlearning.com/2019/WassersteinGAN)

[Open Questions about Generative Adversarial Networks](https://distill.pub/2019/gan-open-problems/)

[Image Completion with Deep Learning in TensorFlow](http://bamos.github.io/2016/08/09/deep-completion/#step-1-interpreting-images-as-samples-from-a-probability-distribution)

[From GAN to WGAN](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)

[A Beginner's Guide to Generative Adversarial Networks (GANs)](https://skymind.ai/wiki/generative-adversarial-network-gan#zero)

[Understanding Generative Adversarial Networks](https://danieltakeshi.github.io/2017/03/05/understanding-generative-adversarial-networks/#fn:goodfellow)

[Generative Adversarial Networks — A Theoretical Walk-Through](https://medium.com/@samramasinghe/generative-adversarial-networks-a-theoretical-walk-through-5889d5a8f2bb)

[Understanding Generative Adversarial Networks (GANs)](https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29)

[An intuitive introduction to Generative Adversarial Networks (GANs)](https://medium.freecodecamp.org/an-intuitive-introduction-to-generative-adversarial-networks-gans-7a2264a81394)

---

# Footnotes and Credits


[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)


---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

