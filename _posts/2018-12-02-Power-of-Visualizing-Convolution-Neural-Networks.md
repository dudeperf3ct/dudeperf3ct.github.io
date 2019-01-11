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

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how to see what a CNN sees?

Well sit tight and buckle up. I will go through everything in-detail.

--insert meme keanu reeves

Feel free to jump anywhere,

- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)




# Regularizations



*In next post, we will discuss about various regularization techniques and when and how to use them. Stay tuned!*

# Introduction to Visualizing CNN

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' />
</p>


<span class='red'>I-know-everything:</span> Today will be the exiciting topic of peeking inside of the black box of CNNs and look at what they see. So, to recap, from our previous posts we saw what a CNN is, wherein we trained CNN and looked at different architectures. Next, we moved to transfer learning. There we learned what an amazing technique transfer learning is! Different ways of transfer learning and how and why transfer learning is providing such a boost. In that topic, we introduced to this notion of CNN as black box, where we really can't tell as to what is that network is looking at while training or predicting and how such amazing CNN learn to classify 1000 categories of 1.2 million images better than humans. So, today we will look behind the scenes of working of CNNs and this will involve looking at lots of pictures. Before that let me tell you a story, once upon a time US Army wanted to use neural networks to automatically detect camouflaged enemy tanks. The dataset that reasearchers collected comprised of 50 photos of camouflaged tanks in trees, and 50 photos of trees without tanks. Using standard techniques of supervisied learning, the reasearchers trained a neural network on the given dataset and achieved an output "yes" for the 50 photos of camouflaged tanks, and output "no" for the 50 photos of forest. The researchers handed the finished work to the Pentagon, which soon handed it back, complaining that in their own tests the neural network did no better than chance at discriminating photos. It turned out that in the researchers' dataset, photos of camouflaged tanks had been taken on cloudy days, while photos of plain forest had been taken on sunny days. The neural network had learned to distinguish cloudy days from sunny days, instead of distinguishing camouflaged tanks from empty forest. Haha! <span class='purple'>The military was now the proud owner of a multi-million dollar mainframe computer that could tell you if it was sunny or not.</span> How much of this urban legend is true or false is compiled in this [blog](https://www.gwern.net/Tanks). Whether happened or not, it sure is a cautionary tale to remind us, to look deep into how neural networks comes to particular conclusion. Along with trust, comes a duty to verify!

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

To start with visualization, we will take a random sample image from our cats and dogs dataset graciously provided by Kaggle. How does this one look?

doggie.png

So, we will performs all sorts of not evil experiments of this dog and using our trained model from transfer learning, we will look at various techniques used for visualizing CNNs.

- Visualizing Activations

In this approach, we look at different filter from our model. From our above architecture, we look at <span class='orange'>block1_conv1</span>, <span class='orange'>block2_conv1</span>, <span class='orange'>block3_conv1</span>, <span class='orange'>block4_conv1</span>, <span class='orange'>block5_conv1</span> and <span class='orange'>block5_conv3</span> layers(filters).

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

- Visualize inputs that maximize the activation of the filters in different layers of the model

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

- Vanilla Backprop

This is one of the simplest of techniques where measure the relative importance of input features by calculating the gradient of the output decision with respect to those input features. It simply means that we use the techniques used above like loss function and calculate the gradients of last layer with respect to model input. The image will we somewhat indiscernible but it shows us what part of image it focuses on to make the output decision.




- Guided Backprop

This method is lot like the one above with the only difference was how to handle the backpropagation of gradients through non-linear layers like ReLU. GuidedBackprop, suppressed the flow of gradients through neurons wherein either of input or incoming gradients were negative. The following illustrations explains clearly this phenomenon.

guided_backprop.png

Here is our result,




- Grad CAM

Grad-CAM, uses the gradients of any target concept (say logits for ‘dog’), flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.


- Guided Grad CAM





- An input that maximizes a specific class


```python
array([[1.7701734e-07, 9.9999988e-01]], dtype=float32)
```

- Deep Dream





<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology


Power of Transfer Learning - Transfer Learning

Power of Visualize CNN - Visualize CNN

ConvNets - Convolution Neural Networks

neurons - unit

---

# Further Reading

[Visualizaing and Understanding Convolution Neural Networks](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)

[Grad CAM](https://arxiv.org/pdf/1610.02391.pdf)

[CS231n Spring 2017 Lecture 11] 

---

# Footnotes and Credits

[Kaggle Dataset for Cats and Dogs](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)

Tanks story sources and great length of discussion  Here: [1](https://www.gwern.net/Tanks) [2](https://www.jefftk.com/p/detecting-tanks) and [3](https://neil.fraser.name/writing/tank/) 

