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

Consider two images, one called content image($$\mathbf{C}$$) and other called style image($$\mathbf{S}$$). The challenge is to grab the styles of style image and grab the content of content image and cut & paste both of them together to get a combined generated image($$\mathbf{G}$$).

-sample_style_transfer

So, from above example, we see that content of content image(left one) is present in combined image(right one). We also see the styles and textures from style image(middle one) to be present in combined image. *Isn't it amazing?*

Now, the question appears how can we extract only content from content image and styles and textures from style image? Extract, this is where we saw CNNs excel at. We saw in our post on [Visualizing CNNs](https://dudeperf3ct.github.io/visualize/cnn/catsvsdogs/2018/12/02/Power-of-Visualizing-Convolution-Neural-Networks/) that different layers extract different patterns like first layers in CNNs extract edges, second layers textures and as we go deep into further layers, high semantic concepts like faces, cars, text, etc are learned. Using these knowledge, we can see that we can use initial layers in CNN to extract styles and the content comes from high layers of CNN. In example below, we can see that if we reconstruct the original image from deeper layers we still preserve the high-level content of the original but lose the exact pixel information.


So, now we got general idea about how using pretrained CNNs can help in extracting patterns, textures and content. But how can we construct new image comprising of these two different representations? This question can be phrased in this way: how can we construct new image such that the content does not differ much from content image and also the new generated image does not differ much in style and texture of style image. Now, this question can be easily solved by creating two loss function: **content loss** ($$\mathcal{L}_{content}$$) and **style loss** ($$\mathcal{L}_{style}$$).

$$\mathcal{L}_{content}(\mathbf{C}, \mathbf{G}) = 0$$ which means we have a loss function which tends to 0 when it's two input images (C and G) are very close to each other in terms of content, and grows as content deviate.

$$\mathcal{L}_{style}(\mathbf{S}, \mathbf{G}) = 0$$ which tells how close in style two input images are to one another.


### Feed-forward Style Transfer


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

---
**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

