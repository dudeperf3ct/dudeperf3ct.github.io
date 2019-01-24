---
layout:     post
title:      Magic of Style Transfer
date:       2018-12-23 12:00:00
summary:    This post will provide an brief introduction to different neural style transfer methods with some examples.
categories: style transfer
published : false
---


# Neural Style Transfer

In this notebook, we will try to answer the question "Can we paint as good as Master Picasso?"

> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Style%20Transfer/style_transfer_keras.ipynb), [PyTorch](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Style%20Transfer/style_transfer_pytorch.ipynb) and [fastai](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Style%20Transfer/style_transfer_fastai.ipynb).  

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.

-meme

Feel free to jump anywhere,
- [Regularization](#regularization)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)




# Regularization


*In next post, we will discuss about other regularization techniques and when and how to use them. Stay tuned!*

# Introduction to Neural Style Transfer

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' />
</p>


<span class='red'>I-know-everything:</span> Today we will do something creative. We will paint (*not literally*). To do this we will use two of our old powers, [Power of Transfer Learning](https://dudeperf3ct.github.io/transfer/learning/catsvsdogs/2018/11/20/Power-of-Transfer-Learning/) and [Force of CNNs](https://dudeperf3ct.github.io/cnn/mnist/2018/10/17/Force-of-Convolutional-Neural-Networks/). I hope you have practised your powers well enough! In [previous post](https://dudeperf3ct.github.io/visualize/cnn/catsvsdogs/2018/12/02/Power-of-Visualizing-Convolution-Neural-Networks/#deep-dream), we encountered how neural networks were dreaming (hallucinating). 

<span class='red'>I-know-nothing:</span> Does this mean we will paint as good as Master Picasso?

<span class='red'>I-know-everything:</span> Haha, I will let you be the judge of that. It's fascinating how this works and the results we obtain. I will directly start with multiple ways of how we can achieve such startling results.

### Artistic Style Transfer 

The [excellent paper](https://arxiv.org/pdf/1508.06576.pdf) in 2015 by Gatys et al proposed a neural algorithm to creates artistic images of high perceptual quality. Let's breakdown how the algorithm creates high quality results. 

Consider two images, one called content image($$\mathbf{C}$$) and other called style image($$\mathbf{S}$$). The challenge is to grab the styles of style image and grab the content of content image and cut & paste both of them together to get a combined pastiche image($$\mathbf{P}$$).

-sample_style_transfer

So, from above example, we see that content of content image(left one) is present in combined image(right one). We also see the styles and textures from style image(middle one) to be present in combined image. *Isn't it amazing?*

Now, the question appears how can we extract only content from content image and styles and textures from style image? Extract, this is where we saw CNNs excel at. We saw in our post on [Visualizing CNNs](https://dudeperf3ct.github.io/visualize/cnn/catsvsdogs/2018/12/02/Power-of-Visualizing-Convolution-Neural-Networks/) that different layers extract different patterns like first layers in CNNs extract edges, second layers textures and as we go deep into further layers, high semantic concepts like faces, cars, text, etc are learned. Using these knowledge, we can see that we can use initial layers in CNN to extract styles and the content comes from high layers of CNN. In example below, we can see that if we reconstruct the original image from deeper layers we still preserve the high-level content of the original but lose the exact pixel information.

--image_reconstruction.png

So, now we got general idea about how using pretrained CNNs can help in extracting patterns, textures and content. But how can we construct new image comprising of these two different representations? This question can be phrased in this way: how can we construct new image such that the content does not differ much from content image and also the new generated image does not differ much in style and texture of style image. Now, the question can be easily solved by creating two loss function: **content loss** ($$\mathcal{L}_{content}$$) and **style loss** ($$\mathcal{L}_{style}$$).

$$\mathcal{L}_{content}(\mathbf{C}, \mathbf{P}) = 0$$ which means we have a loss function which tends to 0 when it's two input images (C and G) are very close to each other in terms of content, and grows as content deviate.

$$\mathcal{L}_{style}(\mathbf{S}, \mathbf{P}) = 0$$ which tells how close in style two input images are to one another.

This style and content transfer problem is to find an image $$\mathbf{P}$$  that differs as little as possible in terms of content from the content image  $$\mathbf{C}$$ , while simultaneously differing as little as possible in terms of style from the style image $$\mathbf{S}$$. In other words, we’d like to simultaneously minimise both the style and content losses.

$$
\begin{aligned}
\mathcal{L}_{total} = \alpha \mathcal{L}_{content} + \beta \mathcal{L}_{style}
\end{aligned}
$$

$$\alpha$$ and $$\beta$$ are numbers that control how much we want to emphasize the content relative to style.

This took us back to MNIST where we had a loss function defined, and then we ran optimizer to minimize the loss. Similarly, Style Transfer is essentially an optimization problem where we minimize $$\mathcal{L}_{total}$$ function. We still haven't figured out as to what these respective loss functions represent(or are defined). To extract various features we use pretrained VGG16 as our base model (there is no standard as to which architecture to prefer). Here is the architecture,

--vgg16

- Content Loss($$\mathcal{L}_{content}$$)

Content Loss is  the (scaled, squared) Euclidean distance between feature representations of the content and combination images. Given a chosen content layer $$\ell$$, let $$\mathbf{F}^\ell$$ be the feature map of our content image $$\mathbf{C}$$ and $$\mathbf{P}^\ell$$ the feature map of our generated pastiche image $$\mathbf{P}$$. The content loss the will be,

$$
\begin{aligned}
\mathcal{L}_{content} = \frac{1}{2} \sum_{i, j}^{}(\mathbf{F}^\ell -  \mathbf{P}^\ell)^2
\end{aligned}
$$

When the content representation of $$\mathbf{C}$$ and $$\mathbf{P}$$ are exactly the same this loss becomes 0.

- Style Loss($$\mathcal{L}_{style}$$)

This loss function is bit tricky. First we define something called Gram Matrix ($$\mathbf{G}$$). Gram Matrix extracts a representation of style by looking at the spatial correlation of the values within a given feature map. If the feature map is a matrix $$\mathbf{F}$$, then each entry in the Gram matrix $$\mathbf{G}$$ can be given by:

$$
\begin{aligned}
\mathbf{G}_{ij} = \sum_{k}^{}\mathbf{F}_{ik} \mathbf{F}_{jk}
\end{aligned}
$$

--gram_matrix.png

If we had two images whose feature maps at a given layer produced the same Gram matrix we would expect both images to have the same style, but not necessarily the same content.

Given a chosen style layer $$\ell$$, the style loss is defined as the euclidean distance between the Gram matrix $$\mathbf{G}^\mathbf{l}$$  of the feature map of our style image $$\mathbf{S}$$ and the Gram matrix $$\mathbf{A}^\mathbf{l}$$  of the feature map of our generated image $$\mathbf{P}$$. When considering multiple style layers we can simply take the sum of the losses at each layer.

$$
\begin{aligned}
\mathcal{L}_{style} = \frac{1}{2} \sum_{\ell=0}^{L}(\mathbf{G}_{ij}^\ell -  \mathbf{A}_{ij}^\ell)^2
\end{aligned}
$$

- Total variation loss

If you were to solve the optimisation problem with only the two loss terms we’ve introduced so far (style and content), we find that the output is quite noisy. We thus add another term, called the [total variation loss](http://arxiv.org/abs/1412.0035) (a regularisation term) that encourages spatial smoothness.

Now final loss function $$\mathcal{L}_{total}$$

$$
\begin{aligned}
\mathcal{L}_{total} = \alpha \mathcal{L}_{content} + \beta \mathcal{L}_{style} + \gamma \mathcal{L}_{TV} 
\end{aligned}
$$

image_generation.gif

All pieces are in place. We run [L-BFS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) or Adam Optimizer (L-BFS is preferred) as an optimizer to minimize the loss function. Et voilà !  the results, 


--results




### Feed-forward Style Transfer

The drawback from above approach, other than computuationally expensive is that we can style only one image at a time. For every other image we have to run the algorithm again.  

The paper, titled [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf) by Johnson et. al, shows that it is possible to train a neural network to apply a single style to any given content image with a single forward pass through the network in real-time and transform any given content image into a styled version. 

--fast_style_transfer.png

Architecture above, contains Image Transform Network and Loss Network.

- Image Transformation Network

The architecture of Image Transfer Net as proposed by Johnson et al is shown in the diagram below.

--image_transform_net

It consists of 3 layers of Conv and ReLU non-linearity, 5 residual blocks, 3 transpose convolutional layers and finally a non-linear tanh layer which produces an output image.

- Loss Network

The loss network is used to calculate a loss between our generated output image and our desired content and style images. We calculate loss in the same way as the previous method, by evaluating the content representation of $$\mathbf{C}$$ and the style representation of $$\mathbf{S}$$ and taking the distance between these and the content and style representations of our output image $$\mathbf{P}$$. These representations are calculated using pretrained VGG16 network.

The training regime consists of a input of content image batch, where ITN transforms it into pastiche images, computes losses using pretrained VGG16 as done above, and calls backward on the final loss to update the ITN parameters. The loss network remains fixed during the training process. In their paper, Johnson et. al trained their network on the [Microsoft COCO dataset](http://mscoco.org/) - which is an object recognition dataset of 80,000 different images.

After training, generating style transfer for any content image takes less than 5 seconds to produced a styled version of given content image. This methods is very fast and efficient than the one above 



<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology


Power of Transfer Learning - Transfer Learning

Power of Visualize CNN - Visualize CNN

ConvNets - Convolution Neural Networks

neurons - unit

cost function - loss or objective function

---

# Further Reading

[A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)

[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)

[A Learned Representation for Artistic Style](https://arxiv.org/pdf/1610.07629.pdf)

[Exploring the structure of a real-time, arbitrary neural artistic stylization network](https://arxiv.org/pdf/1705.06830.pdf)

[Artistic Style Transfer](https://harishnarayanan.org/writing/artistic-style-transfer/)

[PyImageSearch: Neural Style Transfer OpenCV](https://www.pyimagesearch.com/2018/08/27/neural-style-transfer-with-opencv/)


---

# Footnotes and Credits


[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)

[Cat](https://www.yooying.com/p/1941588927056054965_1044254472)

[Lake](https://www.yooying.com/p/1921170375916530754)

---
**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

