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

<span class='red'>I-know-everything:</span> Consider the example where are we are teaching a model to distinguish between dog(y=0) and cat(y=1). The way traditional machine learning classification algorithms like logistic regression or perceptron algorithm tries to find a straight line - a decision boundary - such that when a test image of dog is passed, model checks on which side of decision boundary does the animal falls and makes prediction accordinly. This is what we have learned so far from our past journey, nothing new. But consider a different approach. Instead of learning a decision boundary, what if network learns a model of dog looking at lots of different images of dog and same with cat, a different model of what a cat looks like. Now, when we pass a test image of dog to classify, the test image is matched against the model of dog and model of cat to make prediction. It's like opposite instead of predicting what class it belongs we predict the features from given image. These featuers will tell us who close it resembles to a dog or cat. Algorithms like logistic regression and perceptron learns mappings directly from space of inputs $$\chi$$ to labels {0, 1} i.e. p(y|x) where y $$\in$$ {0, 1} and these are called <span class='purple'>discriminative learning algorithms.</span> Now, instead of learning the mapping from input space, what if model learns the distribution of input features? This is the idea behind <span class='purple'>generative learning algorithms.</span> They learn p(x|y), for example, if y=0 indicates it's a dog then p(x|y=0) models the distribution of dog's features and p(x|y=1) modelling the distribution of cat's features. GANs belong to the family of generative models. This means that GANs samples data from training set, a distribution of $$p_{data}$$, and learns to represent an estimate of that distribution, resulting in probability distribution $$p_{model}$$. There are cases where model estimates $$p_{model}$$ explicitly and in other cases model is only able to generate samples from $$p_{model}$$. <span class='red'>GAN primarily focuses on the second case generating samples from the model distribution</span> although it is possible to design GANs that can do both.

<p align="center">
<img src='/images/gan/sample_gan.jpg' width="50%"/> 
</p>

Here is an example, where an ideal generative model would be able to train on examples shown on left and then create more examples from the same distribution as shown on the right.

<span class='green'>I-know-nothing:</span> Discriminative learning are fairly straight forward, we get data and labels and we train and get state-of-the-art results. I wonder how are generative models trained and why have they not yet been used as first go-to models?

<span class='red'>I-know-everything:</span> The information we gained from above discussion about generative models is that they are about comparing $$p_{model}$$ which is data distribution learned by model with $$p_{data}$$ which is true data distribution. Let's see an example.

<p align="center">
<img src='/images/gan/generative_model.svg' width="50%"/> 
</p>

In the example above, the blue region shows the true data distribution ($$p_{data}$$), where black dot represents each image in dataset. Now our model, a neural network in yellow draws points from unit Gaussian, red in color, and generates a distribution as shown in green color which is the distribution learned by model ($$p_{model}$$ or $$\hat{p_{\theta}}$$). Our goal then is find parameters $$\theta$$ of model that produce a distribution that closely matches the true data distribution. Therefore, you can imagine the green distribution starting out random and then the training process iteratively changing the parameters $$\theta$$ to stretch and squeeze it to better match the blue distribution. There are many loss function as in case of supervised learning which deal with comparing two distribution such as Kullback-Liebler (KL) divergence, Reverse-KL divergence and Jenson-Shannon Divergence (JSD). They belong to F-divergence class of probability distance metrics. The other class is Integral Probability Metrics (IPMs). For the IPMS, we have the Wassterstein distance(which is used in the WGAN) and the Maximum Mean Discrepancy (MMD). Difference between F-divergence and IPMs is F-divergences determine distance using division of two probability distributions, $$\frac{P(x)}{Q(x)}$$ and IPMs use the difference, P(x) - Q(x).

<p align="center">
<img src='/images/gan/distances.png' width="50%"/> 
</p>

Most generative models have this basic setup, but differ in the details. Also, [GANs and Divergence Minimization](https://colinraffel.com/blog/gans-and-divergence-minimization.html) blog by Colin explains F-divergence class through amazing visualizations.

<span class='green'>I-know-nothing:</span> The approach taken by GANs is certainly new compared to our previous approaches of supervised learning. I wonder in what bucket of learning does GAN go in? What's so special about them? What learning function do they use if any?

<span class='red'>I-know-everything:</span> Here's a interesting thing, they belong to both buckets of supervised learning and unsupervised learning. The GAN sets up a supervised learning problem in order to do unsupervised learning. You will understand why so once we introduce different parts of GAN. Let's do that! The basic idea of GAN is setting up game between two players. The two players are <span class='saddlebrown'>generator</span> and <span class='saddlebrown'>discriminator</span>. <span class='purple'>The generator creates samples that are intended to come from the same distribution as the training set. The discriminator examines the samples to determine whether they are real or fake as in, does the input samples belong to the training set or not?</span> So, what is the game between the two players? The generator is trained to fool the discriminator i.e. generator generates a sample and passes it to discriminator. The discriminator using traditional supervised learning is trained to classify the input sample in two classes (real or fake), fooling the discriminator means that discriminator will classify the sample generator to be real instead of fake. And hence, the name "Adversarial". Here we can see that generator wants to be good at fooling discriminator and discriminator wants to be good at classifying samples correctly. This corresponds to [Nash Equilibrium](https://www.youtube.com/watch?v=LJS7Igvk6ZM) from Game Theory. Borrowing example of Alice and Bob from Wikipedia, Alice and Bob are in Nash equilibrium if Alice is making the best decision she can, taking into account Bob's decision while his decision remains unchanged, and Bob is making the best decision he can, taking into account Alice's decision while her decision remains unchanged. Likewise, a group of players are in Nash equilibrium if each one is making the best decision possible, taking into account the decisions of the others in the game as long as the other parties' decisions remain unchanged. GAN requires finding the Nash Equilibrium of the game, which is more difficult than optimizing an objective function as done in traditional machine learning.

Okay, let's try to explain this notion of game through more examples. Suppose the generator to be a counterfeiter, trying to make fake money, and the discriminator as being like police, trying to allow legitimate money and catch counterfeit money. To succeed in this game, the counterfeiter must learn to make money that is indistinguishable from genuine money. Another example, where the generator is person signing cheques using fake signatures and discriminator is bank person, trying to identify if the signature is authentic or not. To succeed in passing the cheque, the person needs to produce signature very real to the original such that bank person gets conned into thinking that fake signature is authentic. You get it right?

Now let's dive in-detail into generator and discriminator.


## GAN Framework

- **Discriminator(D)** is a differentiable function, usually a neural network with parameter $$\theta^{(D)}$$. Discriminative network which takes in input, $$\mathbf{x}$$ "real" which comes from training set or output from generator G($$\mathbf{z}$$) "fake". The goal of discriminator is to classify the input from training set as real and the one from generator as fake. Discriminator is shown half of inputs which are real and remaining half as fakes.

- **Generator(G)** is also differential function, another neural network with parameter $$\theta^{(G)}$$. Generative network takes in input $$\mathbf{z}$$, where $$\mathbf{z}$$ is sample from some prior distribution, G($$\mathbf{z}$$) yields a sample $$\mathbf{x}$$ drawn from $$p_{model}$$. The goal of generator is to fool discriminator.


<p align="center">
<img src='/images/gan/gan.jpg' width="50%"/> 
</p>

Here is the game which is played in two scenarios. In first scenario, left side of the figure, training examples $$\mathbf{x}$$ are randomly sampled from training dataset and used as input for first player, the discriminator(D). The goal of discriminator(D) is to output the probability that its input is real rather than fake. In first scenario, D($$\mathbf{x}$$) tries to be near 1, classifying it to be a real. In second scenario, inputs $$\mathbf{z}$$ to the generator(G) are sampled from model's prior over latent variables. The discriminator then receives the output from generator(G), G($$\mathbf{z}$$), a fake sample generated by generator. Here, the discriminator(D) tries to make D(G($$\mathbf{z}$$)) near 0, as it is fake sample and generator(G) tries to make D(G($$\mathbf{z}$$)) near 1 to fool discriminator in classifying the fake sample as real. If both models have sufficient capacity, then the Nash equilibrium of this game corresponds to the G($$\mathbf{z}$$) being drawn from the same distribution as the training data, and D($$\mathbf{x}$$) = $$\frac{1}{2}$$ for all $$\mathbf{x}$$. How? We will prove this shortly.

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

From looking at the equations above for $$J^{(D)}(\theta^{(D)}, \theta^{(G)})$$ and figure explaining two scenarios of game, the discriminator decision are accurate when it correctly classifies fake and real samples. In terms of cost function, in first scenario with real samples, D($$\mathbf{x}$$) tries to be near 1, i.e. maximize $$\mathbb{E}_{\mathbf{x} \sim p_{data}}[D(\mathbf{x})]$$. That is, when D($$\mathbf{x}$$) becomes close to 1, $$\mathbb{E}[(D(\mathbf{x}))]$$ becomes close to 0 and when D($$\mathbf{x}$$) tries to be near 1, $$\mathbb{E}[(D(\mathbf{x}))]$$ becomes close to $$-\infty$$.  In second scenario with fake samples, D($$\mathbf{x}$$) tries to be near 0, i.e. maximize $$\mathbb{E}_{\mathbf{x} \sim \mathbf{z}}[1-D(G(\mathbf{z}))]$$. (Question:  Show that in the limit, the maximum of the discriminator objective above is the Jenson-Shannon divergence, up to scaling and constant factors.)

The generator on other hand is trained to increase the chances of D producing a high probability i.e. 1, to classify it as real, for a fake example, i.e. maximizing $$\mathbb{E}_{\mathbf{z}}[D(G(\mathbf{z}))]$$ or to minimize $$\mathbb{E}_{\mathbf{z}}[1-D(G(\mathbf{z}))]$$, the part of cost function ($$\mathbb{E}_{\mathbf{x} \sim p_{data}}[D(\mathbf{x})]$$) which deals with real samples will have no effect on generator as it is not sampled from generator.

So, combining both the conclusions from above, <span class='green'>to maximize the cost function for D and minimze the second part of cost function for G, G and D are essentially playing minmax game.</span>

We substitute V(D, G) = - $$J^{(D)}(\theta^{(D)}, \theta^{(G)})$$ in cost function to get the minmax of value function as follows,

$$
\begin{aligned}
\min_{G} \max_{D} V(D, G) &= \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}[\log_{}D(\mathbf{x})]+ \mathbb{E}_{\mathbf{z} \sim p_{z}(\mathbf{z})}[\log_{}(1-D(G(\mathbf{z})))]
\end{aligned}
$$

<span class='blue'>It's like generator and discriminator are fighting each other on who will win.</span> Each wants to complete it's own objective. This game continues till we get a state, in which each model becomes an expert on what it is doing, the generative model increases its ability to get the actual data distribution and produces data like it, and the discriminative becomes expert in identifying the real samples, which increases the system’s classification task. The discriminator tries to maximize tweaking only it's parameter and G tries to minimze tweaking only it's parameters. How amazing? And this setup helps G to produce jaw-dropping images.  Can it get any better than this? Question for curious readers is will doing maxmin produce same results?

<span class='saddlebrown'>On a sad note, the cost used for the generator in the minimax game is useful for theoretical analysis, but does not perform especially well in practice. In the minimax game, the discriminator minimizes a cross-entropy, but the generator maximizes the same cross-entropy. This is unfortunate for the generator, because when the discriminator successfully rejects generator samples with high confidence producing a perfect discriminator, the generator’s gradient vanishes, it will produce zero everywhere, leading to vanishing gradient problem.</span>

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

Different cost functions such as feature matching, minibatch discrimination, etc produces good results in GANs. Many such cost functions can be tried depending on the task at hand and not limited to above. 


## Therotical Limits

We claimed above that after several steps of training, if G and D have enough capacity, they will reach a point at which both cannot improve because $$p_{g}$$ = $$p_{data}$$. The discriminator is unable to differentiate betweenthe two distributions, i.e.D(x) = $$\frac{1}{2}$$.

### Optimal D

We want to find best or the optimal value for D, i.e. D* for fixed G. So, we have cost function, 

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

### Optimal G

We want to 


### Global Optimal

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

Having defined both discriminator (a classifier that takes in input as image and outputs a scalar 1 or 0 depending on input is real or fake), and generator (a neural network that takes in input random noise and produces an image). The next step is to sample minibatch m, first minibatch of m noise samples and second minibatch of m examples from dataset. Then we pass the minibatch of samples containing noise through G to obtain minibatch size of fake images. Next, we train discriminator first on real images whose labels are 1 as they are drawn from true distribution of dataset and then train the same D on fake sample produced from previous step and here pass the labels as 0 as they are fake. Then we calculate the total loss of D which is sum of both losses produced above. Then we set keep D's parameters fixed and pass the minibatch of m samples to G and the fake sample generated whose parameters are trainable are passed to D. But here's the catch. This time we set the labels of these samples as 1, fooling the D, such that they should be classified as real. This way D is guiding G telling it how to tweak it's weights so as to produce good example such that D is fooled. And this process continues for a lot many training epochs.

## Problem in Training GANs

Of course, the training procedure we described above is very unstable and difficult. I mean Is D doing good job in classifying?, Is G generating good samples?, How long should I train to get good examples?, 
 

Walking on the manifold (latent space) that is learnt can usually tell us about signs of memorization (if there are sharp transitions)and about the way in which the space is hierarchically collapsed. If walking in this latent space results in semantic changes to the image generations (such as objects being added and removed), we can reason that the model has learned relevant and interesting representations.


## Recap

Okay, let's breathe for a moment and compress everything in few lines if we can!

- 


## Different types of GANs

GAN literature is filled (overflowing) with different types of GANs or anynameGAN across different domains. We will take a peek into some of the GANs. As we will see each different types of GANs, we will observe how they vary from standard GANs which is similar to ways we discussed what generative models are and what makes GAN different from other types of generative models.

# Images

### DCGAN

DCGAN stands for "Deep Convolution GAN". LAPGAN [paper](https://arxiv.org/pdf/1506.05751) developed an alternative approach to iteratively scale low resolution generated images give that CNN had not great success to provide great image outputs with GAN in previous attempts. The authors of [DCGAN](https://arxiv.org/pdf/1511.06434) also after exploring several models, identified a family of CNN architectures which train GAN stably and generate high quality images. For that to achieve, they proposed 3 major changes to CNN architecures. First, replace all pooling functions with strided convolutions for D and fractional convolutions for G, allowing the network to learn its own spatial downsampling. Second, get rid of any fully-connected layers in both G and D CNN architectures. Third, use batchnorm in both G and D, which stabilizes model learning by normalizing input to each unit to have zero mean and unit variance. Also, using ReLU as activation for all layers in G with exception of output which uses tanh as activation and using LeakyReLU as activation for all layers in D. Authors also use GAN as feature extractor and use it for classifying CIFAR-10 dataset and achieve accuracy of 82% which is about 2% less than CNN.  

In short, make some changes to original GAN architecture and boom better results than standard GAN.

### Results

First result compars DCGAN samples with GAN samples, where DCGAN achieves error rate of 2.98% on 50K samples and GAN achieves 6.28% error rate.

<p align="center">
<img src='/images/gan/dcgan_res1.png' width="50%"/> 
</p>

The second most interesting result obtained from paper is, we can perform arithmetic on images to obtain meaningful representation. For e.g. if we take smiling woman, subtract neutral woman and add netural man, we get smiling man as output. Another one is man with glasses - man without glasses + woman without glasses = woman with glasses. Amazing right? The same we saw in case of [word vectors](), remember?


<p align="center">
<img src='/images/gan/dcgan_res2.png' width="50%"/> 
<img src='/images/gan/dcgan_res3.png' width="40%"/>
</p>

This results walks through the latent space to see if model has not simply memorized training sample. In first row, we see a room without a window slowly transforming into a room with a giant window and in last row, we see what appears to be a TV slowly being transformed into a window.

<p align="center">
<img src='/images/gan/dcgan_res4.png' width="50%"/> 
</p>

### WGAN

The generative models to make the model's distribution close to data distribution either by optimizing distribution using maximum likelihood (Question: Prove that this is equal to minimizing KL divergence.) or learn a function that transforms existing Z (latent variable) into model's distribution. Authors of the [paper](https://arxiv.org/pdf/1701.07875.pdf) propose a different distance metrics to measure the distance between distributions i.e d($$P_{data}$$, $$P_{model}$$). We have seen that there are many other ways to measure the closeness of distribution like KL-divergence, Reverse KL-divergence, Jenson-Shannon(JS) divergence for generative model but each of the above methods don't really converge for some sequence of distribution. (We haven't provided any formal definition of each of method above and leave it as exercise to explore.) Hence, bring in the Earth Mover(EM) distance or Wasserstein-1. [Alex Irpan](https://www.alexirpan.com/2017/02/22/wasserstein-gan.html) provides a great overview of WAN's in general and are great starting point before heading to paper. The intution behind the EM distance is we want our model's distribution $$P_{model}$$ to move close to $$P_{data}$$ true data distribution. Moving mass $$\mathbf{m}$$ by distance $$\mathbf{d}$$ requires effort $$\mathbf{m}\dot\mathbf{d}$$. The earth mover distance is minimal effort we need to spend to bring these distributions close to each other. Authors prove why Wasserstein distance is more compelling than other methods and hence a better fit as loss function for generative models. But Wasserstein distance is intractable in practise. Authors propose alternative approximation which  a result from [Kantorovich-Rubinstein duality](https://en.wikipedia.org/wiki/Wasserstein_metric#Dual_representation_of_W1). [Sebastion Nowozin](https://www.youtube.com/watch?v=eDWjfrD7nJY) provides very excellent introduction to each of the obscure terms above. Here is WGAN algorithm, 

<p align="center">
<img src='/images/gan/wgan.png' width="50%"/> 
</p>

Notice, there is no discriminator and there is something extra term of clipping in the algorithm. Also, we train critic for more time $$n_{critic}$$ times more than generator. The discriminator in GAN is known as critic in WGAN because the critic here is not classifier of real and fake but is trained on Wasserstein loss to output unbounded real number. $$\mathbf{f_{w}}$$ doesn't give output {0, 1} and that is reason why authors call it critic rather than discriminator. Since the loss for the critic is non-stationary, momentum based methods seemed to perform worse. Hence algorithm uses RMSProp instead of Adam as WGAN training becomes unstable at times when one uses a momentum based optimizer. One of the benefits of WGAN is that it allows us to train the critic till optimality. The better the critic,the higher quality the gradients we use to train the generator. This tells us that we no longer need to balance generator and discriminator’s capacity properly unlike in standard GAN.

In short, take GAN change training procedure a little and replace cost function in GANs with Wasserstein loss function.

After 19 days of proposing WGAN, the authors of paper came up with improved and stable method for training GAN as opposed to WGAN which sometimes yielded poor samples or fail to converge. In this method, authors get rid of use of clipping the weights of critic in WGAN and use a different method which is to penalize the norm of gradient of the critic with respect to its input. This new loss is WGAN-GP. 

In short, take GAN change training procedure a little and replace cost function in GANs with WGAN-GP loss function i.e. add gradient penalty term to the previous critic loss.

### Results

After training in LSUN dataset, here are the results produced. Left from WGAN with DCGAN architecture and right from DCGAN.

<p align="center">
<img src='/images/gan/wgan_res1.png' width="50%"/> 
<img src='/images/gan/wgan_res2.png' width="40%"/>
</p>

The result below has on left side WGAN with DCGAN architecture and DCGAN on right with both not using batch norm. If we remove batch norm from the generator, WGAN still generates okay samples, but DCGAN fails completely.

<p align="center">
<img src='/images/gan/wgan_res3.png' width="50%"/> 
<img src='/images/gan/wgan_res4.png' width="40%"/>
</p>

Comparing WGAN on left with standard GAN. GAN suffers from mode collapse. This is the phenomenon that after learning for few epochs dataset, the model goes to failure mode and stops learning. The model starts producing same number of images everywhere with very tiny variations.

<p align="center">
<img src='/images/gan/wgan_res5.png' width="50%"/> 
<img src='/images/gan/wgan_res6.png' width="40%"/>
</p>

The comparison of results of WGAN with WGAN-GP, DCGAN and LSGAN on LSUN dataset,

<p align="center">
<img src='/images/gan/wgan_gp_res.png' width="50%"/> 
</p>




### Pix2Pix

The [researchers](https://arxiv.org/pdf/1611.07004.pdf) at BAIR laboratory devised a method for image to image translation using conditional adversarial networks. The figure below clearly shows what's going on. 

<p align="center">
<img src='/images/gan/pix2pix.png' width="50%"/> 
</p>

Here we model learns to map edges -> photo. The discriminator D, learn to classify between fake(produced by G) and real {edge, photo} tuples. The generator G, learns to fool D. The only difference with previous approach of standard GAN is using conditional GAN. In case of standard GAN, we generator learns mapping from random noise z to output image y, i.e. G : z -> y. In contrast, connditional GANs learns a mapping from observed image x, random noise z to output image y, i.e. G : {x, z} -> y. The new loss function to optimize then becomes,  $$\mathcal{L}_{cGAN} = \mathbb{E}_{\mathbf{x,y}}[\log_{}(D(x,y)] + \mathbb{E}_{\mathbf{x,z}}[\log_{}(1-D(x, G(\mathbf{x, z})))]$$, which is again minmax game G minimizing and D maximizing this objective function.

In short, instead of using standard GAN we use variant called cGAN and accordinly new objective function.

### Results

Paper showed some of the fantasic results obtained by using cGANs.

This figure shows how different domains like segmentation, aerial mapping, colorization, etc can be learned using cGANs.

<p align="center">
<img src='/images/gan/pix2pix_res1.png' width="50%"/> 
</p>

Applying cGANs in domain of semantic segmentation, the result obtained from L1 + cGANs are better than other approaches.

<p align="center">
<img src='/images/gan/pix2pix_res2.png' width="50%"/> 
</p>

This figure shows input, output i.e. {aerial, map} and {map, aerial} tuples which can both be learned by using cGANs.

<p align="center">
<img src='/images/gan/pix2pix_res3.png' width="50%"/> 
</p>

This figure shows the result after applying cGAN for colorization along with results from other approaches.

<p align="center">
<img src='/images/gan/pix2pix_res4.png' width="50%"/> 
</p>

This figure shows uses cGAN for image completion.

<p align="center">
<img src='/images/gan/pix2pix_res6.png' width="50%"/> 
</p>

This shows how to convert sketch to image resembling the sketch. Also, use of cGANs to remove background and transferring of pose in "Do as I do" example shown below.

<p align="center">
<img src='/images/gan/pix2pix_res5.png' width="50%"/> 
</p>


### CycleGAN

Above we visited pix2pix method where we provided pairs input and output to cGAN to learn mapping. [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf) on other hand performs same task in unsupervised fashion without paired examples of transformation from source to target domain. The trick used by CycleGAN that makes them get rid of expensive supervised label in target domain is to double mapping i.e. two-step transformation of source domain image - first by trying to map it to target domain and then back to the original image. Hence, we don't need to explicitly give target domain image. The goal in CycleGAN is to learn the mapping from G : X -> Y such that distribution of images from G(X) is indistinguishable from from the distribution of images of Y. But because this mapping is under-constrained (or not guided), we couple it with an inverse mapping F : Y -> X where we converted the generated image from above mapping back to original image and introduce a cycle consistency loss to enforce F(G(X)) $$\approx$$ X and G(F(Y)) $$\approx$$ Y. Combining this loss along with individual losses of G and F, we get the full objective for unpaired image-to-image translation. This is so good that we will repeat again with the figure below.

<p align="center">
<img src='/images/gan/cyclegan.png' width="50%"/> 
</p>

There are two generators(mapping functions) G : X -> Y and F : Y -> X, and two discriminators $$D_{X}$$ which aims to distinguish images of X(real) & F(Y)(fake) samples and $$D_{Y}$$ which aims to distinguish images of Y(real) & G(X)(fake) samples. $$D_{Y}$$ encourages G to translate X into outputs indistinguishable to Y, and similarly $$D_{X}$$ encourages F to translate Y into outputs indistinguishable to X. The (b) and (c) part are forward and backward cycle-consistency loss introduced to capture the intuition that if we translate from one domain to the other and back again we should arrive at where we started. Forward cycle-consistency in (b) is x -> G(x) -> F(G(x)) $$\approx$$ x and backward cycle-consistency in (c) is y -> F(y) -> G(F(y)) $$\approx$$ y. If we want to write the total loss mathematically, 

$$
\begin{aligned}
\mathcal{L}_{GAN}(G, D_{Y}, X, Y) &= \mathbb{E}_{\mathbf{y} \sim p_{data}(y)}[\log_{}(D_{Y}(y)] + \mathbb{E}_{\mathbf{x} \sim p_{data}(x)}[\log_{}(1 - D_{Y}(G(x)))] \\
\mathcal{L}_{GAN}(F, D_{X}, Y, X) &= \mathbb{E}_{\mathbf{x} \sim p_{data}(x)}[\log_{}(D_{X}(x)] + \mathbb{E}_{\mathbf{y} \sim p_{data}(y)}[\log_{}(1 - D_{X}(F(y)))] \\
\mathcal{L}_{cyc}(G, F) &= \mathbb{E}_{\mathbf{x} \sim p_{data}(x)} ||F(G(x)) - x|| + \mathbb{E}_{\mathbf{y} \sim p_{data}(y)} ||G(F(y)) - y|| \\
\mathcal{L}(G, F, D_{X}, D_{Y}) &= \mathcal{L}_{GAN}(G, D_{Y}, X, Y) + \mathcal{L}_{GAN}(F, D_{X}, Y, X) + \lambda\mathcal{L}_{cyc}(G, F)
\end{aligned}
$$

In short, cycle GAN is another unsupervised learning variant of standard GAN where we learn to translate images from source to target domain. 


### Results

This paper produced most amazing results. Just keep watching.

Horse -> Zebra, really?

<p align="center">
<img src='/images/gan/horse2zebra.gif' width="50%"/> 
</p>

Image-to-image translation can be done in many ways. For example, turning winter images to summer and vice versa, turning horses to zebras and vice versa, turning any photo into Monet style and vice versa. 

<p align="center">
<img src='/images/gan/cyclegan_res1.png' width="50%"/> 
</p>

Here is result of mapping Monet style paintings into photos. Do they look familiar to something we did previously? Yes, [Neural Style Transfer](https://dudeperf3ct.github.io/style/transfer/2018/12/23/Magic-of-Style-Transfer/).

<p align="center">
<img src='/images/gan/cyclegan_res2.png' width="50%"/> 
</p>

Here is the opposite result of turning photos into different styles of painting like Monet, Van Gogh, etc.

<p align="center">
<img src='/images/gan/cyclegan_res3.png' width="50%"/> 
</p>

Who says we need apple to apple comparison? 

<p align="center">
<img src='/images/gan/cyclegan_res4.png' width="50%"/> 
</p>

This result shows photo enhancement achieved by mapping snaps from smartphone to the ones taken on DSLR.

<p align="center">
<img src='/images/gan/cyclegan_res5.png' width="50%"/> 
</p>

### ProGAN

Generating images from 32x32 upto 128x128 with all the new fancy losses seemed cool but generating images of large resolution say 512x512 remained a challenge. The problem with large resolution is that large size implies small minibatches which in turn lead to training instability. We have already visited how training GANs can lead to mode collapse where every output of gan is some number of same images where discriminator wins and generator loses and it's game over. These all problems are the reason why GANs cannot achieve high quality even if we try to make GANs deeper or bigger. The team at Nvidia tackled this challenge through new GANs called ProGAN and bunch of other tricks. The idea behind ProGAN is we start with low resolution images, and then progressively increase the resolution by adding layers to the networks. What happens is instead of using standard GANs where we would have used deep networks to generate high res from latent code, and as the networks are deep it would have taken a lot of time for G to come up with good high res images as D will be already better in rejecting in these samples. This increase in amount of time can lead to mode collapse as already D is better at what it is doing and G is failing to learn anything as layers are deeper and going from randomly intialized weights of each layer to good weight will take a lot of time, if at all possible. So, instead of using standard GANs, the team at Nvidia came up with something called ProGAN. ProGAN starts with tiny images of size 4x4 images and correspondingly shallow networks. The network is trained with this size for sometime until they are more or less converged, next shallow network corresponding to size 8x8 is added which is again trained till convergence and further 16x16 image size network is added. This continues till sizes upto image resolution of 1024x1024 and after 2 days of training these ProGANs we get amazing results. How would G and D look? They would be mirror of each other. That is, in case of 4x4, G will take latent code and produce 4x4 images and D wil take 4x4 and produce real output number(unbounded), as authors use WGAN-GP as loss instead of real and fake. Let's see how it looks,

<p align="center">
<img src='/images/gan/progan.png' width="50%"/> 
</p>

This is how typical training in ProGAN looks like.

<p align="center">
<img src='/images/gan/progan_train.gif' width="50%"/> 
</p>

ProGAN generally trained about 2–6 times faster than a corresponding traditional GAN, depending on the output resolution.

<p align="center">
<img src='/images/gan/progan_train.png' width="50%"/> 
</p>

Here is a typical architecture of ProGAN shown below. The generator architecture for k resolution follows same pattern where each set of layers doubles the representation size and halves the number of channels and discriminator doing the exact oppposite. The ProGAN uses uses nearest neighbors for upscaling and average pooling for downscaling whereas DCGAN uses transposed convolution to change the representation size.

<p align="center">
<img src='/images/gan/progan_arch.png' width="50%"/> 
</p>

That's a very high level overview, but let's dwell on this a bit because they are so cool! Let's look at one such architecture of ProGAN.

<p align="center">
<img src='/images/gan/progan_one_step.png' width="40%"/>
<img src='/images/gan/progan_one_step_D.png' width="40%"/>
</p>

Look at the architecture G and D on left side, we see that they are exact mirrors of each other. Let's walkthrough upto some kxk resolution and see what happens in detail. First generator starts with producing 4x4 image resolution and passing it to D and all backpropogation of error and learning of G and D takes place until some degree of convergence. So, we trained for only 3 layers in G and 3 layers in D for 4x4 resolution. Next, to generate double the resolution 8x8 image, we add 3 more layer to each side of G and D. Now, all the layers in G and D are trainable. To prevent shocks in the pre-existing lower layers from the sudden addition of a new top layer, the top layer is linearly “faded in”. This fading in is controlled by a parameter α, which is linearly interpolated from 0 to 1 over the course of many training iterations. So, there is no problem of catastrophic forgetting and only new layers are learned from scratch. This reduces the training time. Next time when we add 3 more layers to increase the resolution of size to 16x16, they are faded-in with already present 4x4 and 8x8 blocks and this ways G and D fight each other using WGAN-GP as loss function upto a desired number of resolution.

To further increase the quality of images and variation, authors propose 3 tricks such as pixel normalization(different from batch or layer or adaptive instance normalization), minibatch standard deviation and equalized learning rate. In minibatch standard deviation, D is given a superpower to penalize G if the variation between training images and the once produced by G is high. G will be forced to produce same variation as in training data. To achieve this equalized learning rate, they scale the weights of a layer according to how many weights that layer has using. This makes sure all the layers are updated at same speed to ensure fair competition between G and D. Pixelwise feature normalization prevents training from spiraling out of control and discourages G from generating broken images.

And the last contribution made was how to evaluate two G's, which one is better? This can be done through Sliced Wasserstein Distance (SWD) where we generate large number of images and extract random 7x7 pixels neighbourhood. We interpret these neighbourhood points as in 7x7x3 dimensional space and comparing this point cloud against the real images(same process) point cloud which can be repeated for each scale.

### Results

I will let the results speak for themselves. Remember none of these faces are real. They are synthesized by G.

<p align="center">
<img src='/images/gan/progan_res.png' width="50%"/> 
</p>

After walking the latent space which is continuous, one such output is this. Notice the changes to hairs, expression, shape of face. Amazing!

<p align="center">
<img src='/images/gan/progan_res.gif' width="50%"/> 
</p>


### StyleGAN

ProGAN as pretty mouthful, right? The authors of Nvidia came out with this paper called StyleGAN where we can by modifying the input of each level separately, control the visual features that are expressed in that level, from coarse features (pose, face shape) to fine details (hair color), without affecting other levels. What this means? Let's look at example below and understand what this means. 

<p align="center">
<img src='/images/gan/stylegan.jpg' width="50%"/> 
</p>
<p align="center">
<img src='/images/gan/stylegan_1.jpg' width="50%"/> 
</p>

We are copying the styles from different resolutions of source B to the images from source A. Copying the styles corresponding to coarse spatial resolutions ($$4^{2}$$–$$8^{2}$$) brings high-level aspects such as pose, general hair style, face shape, and eyeglasses from source B, while all colors(eyes, hair, lighting) and finer facial features resemble A. If we instead copy the styles of middle resolutions ($$16^{2}$$–$$32^{2}$$) from B, we inherit smaller scale facial features, hair style, eyes open/closed from B, while the pose, general face shape, and eyeglasses from A are preserved. Finally, copying the fine styles ($$64^{2}$$–$$1024^{2}$$) from B brings mainly the color scheme and microstructure. 

<p align="center">
<img src='/images/gan/stylegan_arch.png' width="50%"/> 
</p>

How does it work then? StyleGANs are upgraded version of ProGAN where we can each progessive layers can be utilized to control different visual features of image. The generator in StyleGAN starts from a learned constant input and adjusts the "style" of the image at each convolution layer based on the latent code, therefore directly controlling the strength of image features at different scales. As we saw from above example, coarse styles can be controlled using $$4^{2}$$–$$8^{2}$$ resolution, middle styles controlled by $$16^{2}$$–$$32^{2}$$ resolution layers and finer styles using $$64^{2}$$–$$1024^{2}$$ resolutions. The typical ProGAN shown on the left side in image above uses progressive layer training to produce high resolution images but StyleGAN uses a different generator approach. Instead of mapping latent code z to resolution, it uses Mapping Network, which maps the latent code z to an intermediate vector w. The latent vector is sort of like a style specification for the image. The purpose of using different mapping netork is suppose we wanted change the hair color of image by nudging the value in latent vector, but what if output produces different gender, or glasses, etc. This is called feature entanglement. The authors state that the intermediate latent space is free from that restriction and is therefore allowed to be disentangled. The mapping network in the paper f shown in images above consists of 8 fully-connected layer and produce w of size 512x1.

<p align="center">
<img src='/images/gan/stylegan_arch_1.png' width="50%"/> 
</p>

We can view the mapping network and affine transformations as a way to draw samples for each style from a learned distribution, and the synthesis network as a way to generate a novel image based on a collection of styles. This synthesis network takes in the output w generated by mapping network to generate image by using AdaIN with whom we had encounter with before [here](https://dudeperf3ct.github.io/style/transfer/2018/12/23/Magic-of-Style-Transfer/#arbitrary-neural-artistic-stylization-network). 

<p align="center">
<img src='/images/gan/stylegan_arch_2.png' width="50%"/> 
</p>





### Results





### BigGAN

The team at Deepmind showed that GANs benefits from scaling and trained models with two to four times as many parameters and eight times the batch size compared to prior art. BigGANs uses class-conditional GANs where they pass class-information to G  and to D using projection as shown below where they pass class information using inner-product with output of D. The objective used by BigGAN is hinge loss. BigGAN adds direct skip-connections from noise vector z to multiple layers of G rather than just initial layer in standard GANs. The intuition behind this design is to allow G to use the latent space to directly influence features at different resolutions and levels of hierarchy. The latent vector z is concatenated with class embeddings and passed to each residual block through skip connections. Skip-z provides a modest performance improvement of around 4%, and improves training speed by 18%. Residual Up used for upsampling in BigGAN G's shown in (b) and Residual Down for downsampling in BigGAN D's is shown in (c). 

<p align="center">
<img src='/images/gan/bigan_project.png' width="40%"/> 
<img src='/images/gan/biggan_arch.png' width="50%"/> 
</p>

BigGAN also employed few tricks such as Truncation Trick, where in previous literature of GAN latent vectors z are drawn from either $$\mathcal{N}$$(0, 1) or $$\mathcal{U}$$[-1, 1]. Instead BigGAN latent vectors are sampled from truncated normal distribution where values which fall outside a range are resampled to fall inside that range. Authors observe that using this sampling strategy does not work well with large models and hence add a Orthogonal Regularization as penalty. One important conclusion drawn from this is that we do not need to use explicit multiscale method as used in ProGAN and StyleGAN for producing higher resolution images. Despite these improvements, BigGAN undergoes training collapse. The authors explore in great-detail why it happens so through colorful plots. They also provide results and conclusion of large amount of experiements performed from which a lot can be learned.

In short, BigGAN could do what ProGAN thought would require multi-scale approach in single-scale by using some tricks.

### Results

Jaw-dropping moment. All the images generated by generator from scratch. Get outta here!

<p align="center">
<img src='/images/gan/biggan_res_0.png' width="50%"/> 
</p>

<p align="center">
<img src='/images/gan/biggan_res_1.png' width="50%"/> 
</p>

[Zaid Alyafeai](https://thegradient.pub/bigganex-a-dive-into-the-latent-space-of-biggan/) provides different tricks we can perform in latent space. 

- **Breeding** : In this we breed between two different classes, i.e we create intermediate classes using a combination of two different classes. 

<p align="center">
<img src='/images/gan/biggan_res_2.png' width="50%"/> 
</p>

- **Background Hallucination** : Here we try to change the background keeping the foreground object the same.

<p align="center">
<img src='/images/gan/biggan_res_3.png' width="50%"/> 
</p>


- **Natural Zoom** : We zoom into certain generated image to look into finer details. It's amazing.

<p align="center">
<img src='/images/gan/biggan_res_4.png' width="50%"/> 
</p>

Here is example of walking in latent space for specific z and c pairs,

<p align="center">
<img src='/images/gan/biggan_res_5.png' width="50%"/> 
</p>


### GAN semi-supervised learning

In paper [Improving GAN by training](https://arxiv.org/pdf/1606.03498.pdf), authors demonstrate they are able to achieve 99.14% accuracy with only 10 labeled examples per class with a fully connected neural network on MNIST dataset. The basic idea of semi-supervised learning with GANs is to use feature matching objective and turn add extra task for discriminator i.e. in addition to classify it will also predict the label of the image. The fake samples of generator can be used as dataset for which discriminator will predict a class corresponding to that image. The feature matching objective is a new objective for G is to train the generator to match the expected value of the features on an intermediate layer of the discriminator. If $$f(\mathbf{x})$$ denote activations on an intermediate layer of the discriminator, then new objective for generator is defined as $$||\mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}[f(\mathbf{x})] - \mathbb{E}_{\mathbf{z} \sim p_{z}(\mathbf{z})}[f(G(\mathbf{z}))]||^{2}_{2}$$. Feature matching is effective in situations where regular GAN becomes unstable.

# Speech

Okay, enough images. Show me(GAN) what else you got. Synthesizing speech is one the cool areas GAN have played a significant role. 



# Text

What else you got? 


# Video

GANs for video has mindblowing applications. Remember deepfakes, the one which everyone is worried about. Yes, it was born here.




## Problems in GANs

How should we evaluate GANs and when should we use them? 
How does GAN training scale with batch size? 
Can we Scale GANs Beyond Image Synthesis? 
Can GANs attain Nash Equilibrium?

## Are we doomed?

You are showing all these cool results with images and videos, the one with the deepfakes, fake speech, etc. This does have serious implications on the society. Are there any counter measures we should be aware of? 

- Images : Worry not if you possess [Art of Observation](https://fs.blog/2013/04/the-art-of-observation/).



Look at the progress, introducing paper in 2014 to worrying about dangerous impact on society caused by GAN in 2017, what can I say more? And this is the story of GANs.

In next post, we will do something <span class='yellow'>different</span>. We will attempt to dissect any one or two papers. Any suggestions? So, let's call that Paper dissection. And further build a text recognizer application and deploy it for fun. A lot to come, a lot of fun!

<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology

GANs - Generative Adversarial Networks

D - Discriminator

G - Generator

z - Latent vector, code or noise vector


---

# Further Reading

[NIPS 2016 Tutorial : Generative Adversarial Network](https://arxiv.org/pdf/1701.00160.pdf)

Fastai [Lesson 12: Deep Learning Part 2 2018 - Generative Adversarial Networks (GANs)](https://www.youtube.com/watch?v=ondivPiwQho&list=PLfYUBJiXbdtTttBGq-u2zeY1OTjs5e-Ia&index=5)

[CVPR 2018 Tutorial on GANs](https://sites.google.com/view/cvpr2018tutorialongans/)

[A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8)

Chapter 3, 5 and 20 of [Deep Learning Book](https://www.deeplearningbook.org/)

[Generative Learning algorithms](http://cs229.stanford.edu/notes/cs229-notes2.pdf)

[Generative Models by OpenAI](https://blog.openai.com/generative-models/#gan)

[Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)

[Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf)

[LAPGAN](https://arxiv.org/pdf/1506.05751)

[DCGAN](https://arxiv.org/pdf/1511.06434.pdf)

[WGAN](https://arxiv.org/pdf/1701.07875.pdf)

Alex Irpan's blog [Read-through: Wasserstein GAN](https://www.alexirpan.com/2017/02/22/wasserstein-gan.html)

[Sebastion Nowozin Generative Adversarial Networks](https://www.youtube.com/watch?v=eDWjfrD7nJY)

Curriculum for learning [Wasserstein GAN from depthfirstlearning](http://www.depthfirstlearning.com/2019/WassersteinGAN)

[CycleGAN](https://arxiv.org/pdf/1703.10593.pdf)

[CycleGAN page](https://junyanz.github.io/CycleGAN/)

[ProGAN](https://arxiv.org/pdf/1710.10196.pdf)

[ProGAN talk by Tero Karras, NVIDIA](https://www.youtube.com/watch?v=ReZiqCybQPA)

[Blog on ProGAN: How NVIDIA Generated Images of Unprecedented Quality by Sarah Wolf](https://towardsdatascience.com/progan-how-nvidia-generated-images-of-unprecedented-quality-51c98ec2cbd2)

[StyleGAN](https://arxiv.org/pdf/1812.04948)

[BigGAN](https://arxiv.org/pdf/1809.11096.pdf)

[BigGanEx: A Dive into the Latent Space of BigGan](https://thegradient.pub/bigganex-a-dive-into-the-latent-space-of-biggan/)

[Are GANs Created Equal? A Large-Scale Study](https://arxiv.org/pdf/1711.10337.pdf)

[Open Questions about Generative Adversarial Networks](https://distill.pub/2019/gan-open-problems/)

[Image Completion with Deep Learning in TensorFlow](http://bamos.github.io/2016/08/09/deep-completion/#step-1-interpreting-images-as-samples-from-a-probability-distribution)

[From GAN to WGAN](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)

[A Beginner's Guide to Generative Adversarial Networks (GANs)](https://skymind.ai/wiki/generative-adversarial-network-gan#zero)

[Understanding Generative Adversarial Networks](https://danieltakeshi.github.io/2017/03/05/understanding-generative-adversarial-networks/#fn:goodfellow)

[Understanding and Implementing CycleGAN in TensorFlow](https://hardikbansal.github.io/CycleGANBlog/)

[Turning Fortnite into PUBG with Deep Learning (CycleGAN)](https://towardsdatascience.com/turning-fortnite-into-pubg-with-deep-learning-cyclegan-2f9d339dcdb0)

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

