---
layout:     post
title:      Power of Visualizing Convolution Neural Networks
date:       2018-12-02 12:00:00
summary:    This post will provide an brief introduction to visualize trained CNN through transfer learning using Dogs vs Cats Redux Competition dataset from Kaggle.
categories: visualize cnn catsvsdogs
published : false
---


# Visualizing CNN

In this notebook, we will try to answer the question "What CNN sees?" using [Cats vs Dogs Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) Competition dataset from kaggle. We will implement this using one of the popular deep learning framework <span class='yellow'>Keras</span> . 

> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Transfer%20Learning/transfer_learning_keras.ipynb), [PyTorch](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Transfer%20Learning/transfer_learning_pytorch.ipynb), and [fastai](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Transfer%20Learning/transfer_learning_fastai.ipynb).  

> *All codes can be run on Google Colab (link provided in notebook).* [Tensorflow implementation](https://github.com/PAIR-code/saliency/blob/master/Examples.ipynb) by PAIR Initiative.

Hey yo, but how to see what a CNN sees?

Well sit tight and buckle up. I will go through everything in-detail.

--insert meme keanu reeves

Feel free to jump anywhere,

- [Introduction to Visualizing CNN](#introduction-to-visualizing-cnn)
  - [Story](#story)
  - [Visualizing Activations](#visualizing-activations)
  - [Visualize inputs that maximize the activation of the filters in different layers of the model](#visualize-inputs-that-maximize-the-activation-of-the-filters-in-different-layers-of-the-model)
  - [Vanilla Backprop](#vanilla-backprop)
  - [Guided Backprop](#guided-backprop)
  - [Grad CAM](#grad-cam)
  - [Guided Grad CAM](#guided-grad-cam)
  - [An input that maximizes a specific class](#an-input-that-maximizes-a-specific-class)
  - [Deep Dream](#deep-dream)
  - [t-SNE Visualization](#t-sne-visualization)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)




# Regularizations



*In next post, we will discuss about various regularization techniques and when and how to use them. Stay tuned!*

# Introduction to Visualizing CNN

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' />
</p>


<span class='red'>I-know-everything:</span> Today will be the exiciting topic of peeking inside of the black box of CNNs and look at what they see. So, to recap, from our previous posts we saw what a CNN is, wherein we trained CNN and looked at different architectures. Next, we moved to transfer learning. There we learned what an amazing technique transfer learning is! Different ways of transfer learning and how and why transfer learning is providing such a boost. In that topic, we introduced to this notion of CNN as black box, where we really can't tell as to what is that network is looking at while training or predicting and how such amazing CNN learn to classify 1000 categories of 1.2 million images better than humans. So, today we will look behind the scenes of working of CNNs and this will involve looking at lots of pictures. 

### Story
Before that let me tell you a story, once upon a time US Army wanted to use neural networks to automatically detect camouflaged enemy tanks. The dataset that reasearchers collected comprised of 50 photos of camouflaged tanks in trees, and 50 photos of trees without tanks. Using standard techniques of supervisied learning, the reasearchers trained a neural network on the given dataset and achieved an output "yes" for the 50 photos of camouflaged tanks, and output "no" for the 50 photos of forest. The researchers handed the finished work to the Pentagon, which soon handed it back, complaining that in their own tests the neural network did no better than chance at discriminating photos. It turned out that in the researchers' dataset, photos of camouflaged tanks had been taken on cloudy days, while photos of plain forest had been taken on sunny days. The neural network had learned to distinguish cloudy days from sunny days, instead of distinguishing camouflaged tanks from empty forest. Haha! <span class='purple'>The military was now the proud owner of a multi-million dollar mainframe computer that could tell you if it was sunny or not.</span> How much of this urban legend is true or false is compiled in this [blog](https://www.gwern.net/Tanks). Whether happened or not, it sure is a cautionary tale to remind us, to look deep into how neural networks comes to particular conclusion. <span class='saddlebrown'>Trust, but verfify!</span>

<span class='green'>I-know-nothing:</span> Ayye Master, I am ready!

<span class='red'>I-know-everything:</span> Before jumping to lots of image, let's look at architecture that was used in training our model.

```python
input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 25088)             0         
_________________________________________________________________
dense_4 (Dense)              (None, 512)               12845568  
_________________________________________________________________
dropout_3 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 512)               262656    
_________________________________________________________________
dropout_4 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 513       
=================================================================
Total params: 27,823,425
Trainable params: 13,108,737
Non-trainable params: 14,714,688
_________________________________________________________________

```
So, it's VGG16 with last fully connected removed with addition of new FC layers along with dropout.

To start with visualization, we will take a random sample image from our cats and dogs dataset graciously provided by Kaggle. How does these ones look?

<p align="center">
<img src='/images/visualize_cnn_files/doggie.png' width="30%"/><img src='/images/visualize_cnn_files/cat_dog.png' width="30%"/>
</p>

So, we will performs all sorts of not evil experiments of this dog and using our trained model from transfer learning, we will look at various techniques used for visualizing CNNs.

### Visualizing Activations

In this approach, we look at different filter from our model. From our above architecture, we look at <span class='orange'>block1_conv1</span>, <span class='orange'>block2_conv1</span>, <span class='orange'>block3_conv1</span>, <span class='orange'>block4_conv1</span>, <span class='orange'>block5_conv1</span> and <span class='orange'>block5_conv3</span> layers(filters). In each layer, some neurons will remain dark while others will light up as they respond to different patterns in the image.

<p align="center">
<img src='/images/visualize_cnn_files/conv1_layer1.png' width="60%"/>
<p align="center">block1_conv1</p>
</p>
<p align="center">
<img src='/images/visualize_cnn_files/conv3_layer1.png' width="60%"/>
<p align="center">block3_conv1</p>
</p>
<p align="center">
<img src='/images/visualize_cnn_files/conv5_layer1.png' width="60%"/>
<p align="center">block5_conv1</p>
</p>
<p align="center">
<img src='/images/visualize_cnn_files/conv5_layer3.png' width="60%"/>
<p align="center">block5_conv3</p>
</p>


Well, this black and white isn't telling much. Let's apply these as a mask to our original image.

<p align="center">
<img src='/images/visualize_cnn_files/conv1_layer1_edge_detector.png' width="25%"/><img src='/images/visualize_cnn_files/conv5_layer1_eyes_nose.png' width="25%"/><img src='/images/visualize_cnn_files/conv5_layer3_ears.png' width="25%"/><img src='/images/visualize_cnn_files/conv5_layer3_nose_whiskers.png' width="25%"/>
</p>


And voila! Wow! We get some amazing results. We see that there are some neurons in the filters which look for whiskers and nose of dog, some on left eye, eyes and nose detector and also ears, etc.

### Visualize inputs that maximize the activation of the filters in different layers of the model

In this approach, we take our trained model and reconstruct the images that maximize the activation of filters in different layers of the model. The reasoning behind this idea is that a pattern to which the unit is responding maximally could be a good first-order representation of what a unit is doing. One simple way of doing this is to find, for a given unit, the input sample(s) (from either the training or the test set) that give rise to the highest activation of the unit. Here, we take our model and layer name for which pattern is to be discerned. Next, we compute the gradient of input image(random noise) with respect to loss function which maximises the activation of provided layer name. We perform a gradient ascent in the input space with regard to our filter activation loss. This along with some tricks like normalizing input, we get beautiful pattern of what pattern that layer activates most for. We could use the same code to display what sorts of input maximizes each filter in each layer. As there are 100s of filters in some layer, we choose the ones with highest activation to form a grid of 8x8.

<p align="center">
<img src='/images/visualize_cnn_files/block1_conv1.png' width="30%"/><img src='/images/visualize_cnn_files/block2_conv1.png' width="30%"/>
</p>
<p align="center">
<img src='/images/visualize_cnn_files/block3_conv1.png' width="30%"/><img src='/images/visualize_cnn_files/block4_conv1.png' width="30%"/>
</p>
<p align="center">
<img src='/images/visualize_cnn_files/block5_conv1.png' width="30%"/><img src='/images/visualize_cnn_files/block5_conv3.png' width="30%"/>
</p>

We get to see some interesting patterns. The first layers basically just encode direction and color. These direction and color filters then get combined into basic grid and spot textures. These textures gradually get combined into increasingly complex patterns. In the highest layers (block5_conv2, block5_conv3) we start to recognize textures. We confirm our claim from previous post on transfer learning, that the lower convolutional layers capture low-level image features, e.g. edges, while higher convolutional layers capture more and more complex details, such as body parts, faces, and other compositional features. We see that this is indeed the case. 

### Vanilla Backprop

This is one of the simplest of techniques where measure the relative importance of input features by calculating the gradient of the output decision with respect to those input features. It simply means that we use the techniques used above like loss function and calculate the gradients of last layer with respect to model input. The image will we somewhat indiscernible but it shows us what part of image it focuses on to make the output decision.

<p align="center">
<img src='/images/visualize_cnn_files/backprop_dog.png' width="30%"/>
<p align="center"> Vanilla Backprop </p>
</p>


### Guided Backprop

This method is lot like the one above with the only difference was how to handle the backpropagation of gradients through non-linear layers like ReLU. GuidedBackprop, suppressed the flow of gradients through neurons wherein either of input or incoming gradients were negative. Also, Guided Backpropagation visualizations were generally less noisy. The following illustrations explains clearly this phenomenon. 

<p align="center">
<img src='/images/visualize_cnn_files/guided_gradcam.png'/>
<p align="center"><a href="https://arxiv.org/pdf/1610.02391.pdf">Image Credits</a></p>
</p>

Here is our result,

<p align="center">
<img src='/images/visualize_cnn_files/guided_backprop_dog.png' width="30%"/>
<p align="center">Guided Backprop of only dog image</p>
</p>


### Grad CAM

Grad-CAM, uses the gradients of any target concept (say logits for ‘dog’), flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept. Given an image and a class of interest (‘dog’) as input, we forward propagate the image through the CNN part of the model and then through task-specific computations to obtain a raw score for the category. The gradients are set to zero for all classes except the desired class (dog), which is set to 1. This signal is then backpropagated to the rectified convolutional feature maps of interest, which we combine to compute the coarse Grad-CAM localization (blue heatmap) which represents where the model has to look to make the particular decision. Here is the illustration from [original paper](https://arxiv.org/pdf/1610.02391.pdf).

Here is the result,

<p align="center">
<img src='/images/visualize_cnn_files/doggie.png' width="20%"/><img src='/images/visualize_cnn_files/cat_dog.png' width="20%"/><img src='/images/visualize_cnn_files/cat_dog.png' width="20%"/>
</p>

<p align="center">
<img src='/images/visualize_cnn_files/grad_cam_dog.png' width="20%"/><img src='/images/visualize_cnn_files/grad_cam_newdog.png' width="20%"/><img src='/images/visualize_cnn_files/grad_cam_newcat.png' width="20%"/>
</p>

<span class='purple'>Amazing right? It tells us exactly what region in the input image it has looked at to make the decision of predicting particular class.</span>


### Guided Grad CAM

Combining Guided Backprop and Grad CAM from above gives Guided Grad-CAM, which gives high-resolution class-discriminative visualizations. It's just pointwise multiplication of above two results.

<p align="center">
<img src='/images/visualize_cnn_files/doggie.png' width="20%"/><img src='/images/visualize_cnn_files/cat_dog.png' width="20%"/><img src='/images/visualize_cnn_files/cat_dog.png' width="20%"/>
</p>

<p align="center">
<img src='/images/visualize_cnn_files/guided_gradcam_dog.png' width="20%"/><img src='/images/visualize_cnn_files/guided_gradcam_newdog.png' width="20%"/><img src='/images/visualize_cnn_files/guided_gradcam_newcat.png' width="20%"/>
</p>


### An input that maximizes a specific class

In this method, we take random noise and with choosing a particular class either cat or dog, we construct an input from random noise such that it gives near perfect accurate prediction of chosen class. 

Here is the noise,

<p align="center">
<img src='/images/visualize_cnn_files/noise.png' width="30%"/>
<p align="center"> Noise </p>
</p>


This is what we obtain as an image.

<p align="center">
<img src='/images/visualize_cnn_files/class_maximization.png' width="30%"/>
<p align="center"> Noise </p>
</p>

This is the output predictions when we pass the image through our model. It is almost certain 99.99% sure that this is dog.(Haha! Well, I don't see how this is a dog.) 

```python
array([[1.7701734e-07, 9.9999988e-01]], dtype=float32)
```
Fchollet on keras blog explains,

> So our convnet's notion of a dog looks nothing like a dog --at best, the only resemblance is at the level of local textures (ears, maybe whiskers or nose). Does it mean that convnets are bad tools? Of course not, they serve their purpose just fine. What it means is that we should refrain from our natural tendency to anthropomorphize them and believe that they "understand", say, the concept of dog, or the appearance of a magpie, just because they are able to classify these objects with high accuracy. They don't, at least not to any any extent that would make sense to us humans. <span class='blue'>So what do they really "understand"? Two things: first, they understand a decomposition of their visual input space as a hierarchical-modular network of convolution filters, and second, they understand a probabilitistic mapping between certain combinations of these filters and a set of arbitrary labels.</span> Naturally, this does not qualify as "seeing" in any human sense, and from a scientific perspective it certainly doesn't mean that we somehow solved computer vision at this point.

### Deep Dream

This is certainly the coolest technique. It's like our neural network is dreaming. In usual CNN training what we do is adjust the network's weight to agree more with the image. But here, we instead adjust the image to agree more with the network. If we adjust the image like this, adjusting the pixels a bit at a time and then repeating, then we would actually start to see dogs in the photo, even if there weren't dogs there to begin with! In other words, instead of forcing the network to generate pictures of dogs or other specific objects, we let the network create more of whatever it saw in the image.

Here is the excerpt from the [blog](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)

> Instead of exactly prescribing which feature we want the network to amplify, we can also let the network make that decision. In this case we simply feed the network an arbitrary image or photo and let the network analyze the picture. We then pick a layer and ask the network to enhance whatever it detected. Each layer of the network deals with features at a different level of abstraction, so the complexity of features we generate depends on which layer we choose to enhance. For example, lower layers tend to produce strokes or simple ornament-like patterns, because those layers are sensitive to basic features such as edges and their orientations.

> If we choose higher-level layers, which identify more sophisticated features in images, complex features or even whole objects tend to emerge. Again, we just start with an existing image and give it to our neural net. We ask the network: “Whatever you see there, I want more of it!” This creates a feedback loop: if a cloud looks a little bit like a bird, the network will make it look more like a bird. This in turn will make the network recognize the bird even more strongly on the next pass and so forth, until a highly detailed bird appears, seemingly out of nowhere.

Here is sample input image,

<p align="center">
<img src='/images/visualize_cnn_files/cinque_terre.jpg' width="60%"/>
<p align="center"> <a href="https://commons.wikimedia.org/w/index.php?curid=32998590"> Cinque Terre Credits </a> </p>
</p>

This is the output we obtain, 

<p align="center">
<img src='/images/visualize_cnn_files/cinque_terre_deepdream.png' width="60%"/>
<p align="center"> Deep Dream Cinque Terre </a> </p>
</p>


We can see lots of dogs in this image. For more hallucinations, check this [results](https://photos.google.com/share/AF1QipPX0SCl7OzWilt9LnuQliattX4OUCj_8EP65_cTVnBmS1jnYgsGQAieQUc1VQWdgQ?key=aVBxWjhwSzg2RjJWLWRuVFBBZEN1d205bUdEMnhB)
.  


### t-SNE Visualization

We will randomly sample 100 images from training set and use penultimate layer as predictor and visualize these 100 images in embedding projector by Tensorflow.


There are other implementations we haven't looked at like occlusion

<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology


Power of Transfer Learning - Transfer Learning

Power of Visualize CNN - Visualize CNN

ConvNets - Convolution Neural Networks

neurons - unit

---

# Further Reading

Best Visualizations ever! on [distill.pub](https://distill.pub/)

Must Read! [Feature Visualization](https://distill.pub/2017/feature-visualization/)

Must Read! [The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/)

Must Read! [Feature-wise transformations](https://distill.pub/2018/feature-wise-transformations/)

[Visualizaing and Understanding Convolution Neural Networks](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)

[Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)

[Guided Backprop](https://arxiv.org/pdf/1412.6806.pdf)

[Grad CAM and Guided Grad CAM](https://arxiv.org/pdf/1610.02391.pdf)

[CS231n Spring 2017 Lecture 11] 

Amazing [PAIR Code Saliency](https://pair-code.github.io/saliency/) Example

[Qure.ai blog on Visualizations](http://blog.qure.ai/notes/deep-learning-visualization-gradient-based-methods)

[Visualizing Higher-Layer Features of a Deep Network](https://www.researchgate.net/publication/265022827_Visualizing_Higher-Layer_Features_of_a_Deep_Network)

[Deep Dream blog by Google](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)

[Amazing keras blog on ConvNet Visualization](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)

[Want to get high?](https://photos.google.com/share/AF1QipPX0SCl7OzWilt9LnuQliattX4OUCj_8EP65_cTVnBmS1jnYgsGQAieQUc1VQWdgQ?key=aVBxWjhwSzg2RjJWLWRuVFBBZEN1d205bUdEMnhB)

---

# Footnotes and Credits

[Kaggle Dataset for Cats and Dogs](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)

Tanks story sources and great length of discussion  Here: [1](https://www.gwern.net/Tanks) [2](https://www.jefftk.com/p/detecting-tanks) and [3](https://neil.fraser.name/writing/tank/) 

[Guided Grad CAM](https://arxiv.org/pdf/1610.02391.pdf) and [Cat and Dog Image](https://arxiv.org/pdf/1610.02391.pdf)

[Cinque Terre Image](https://commons.wikimedia.org/w/index.php?curid=32998590)

