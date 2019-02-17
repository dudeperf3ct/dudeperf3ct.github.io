---
layout:     post
title:      Force of Recurrent Neural Networks
date:       2019-01-19 12:00:00
summary:    This post will provide an brief introduction to recurrent neural networks.
categories: rnn
published : false
---


# Recurrent Neural Networks

In this notebook, we will see if Neural Networks can write as good as Shakespeare?

> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Object%20Detection/Keras/object_detection_keras.ipynb), [PyTorch](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Object%20Detection/PyTorch/object_detection_pytorch.ipynb), [Tensorflow](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Object%20Detection/Tensorflow/object_detection_tensorflow.ipynb), [fastai](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Object%20Detection/Fastai/object_detection_fastai.ipynb) and [Demos](https://github.com/dudeperf3ct/DL_notebooks/blob/master/Object%20Detection/Demos).  

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.

-meme


Feel free to jump anywhere,

- [Introduction to Recurrent Neural Networks](#introduction-to-recurrent-neural-networks)
- [Recap](#recap)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)

Here, let me borrow some of the points from [Prof. Yuval Noah Harrari's](https://www.ynharari.com/about/) book on [Sapiens](https://www.amazon.com/Sapiens-A-Brief-History-Humankind/dp/1846558239/),

> <span class='purple'>How did Homo sapiens came to dominate the planet? The secret was a very peculiar characteristic of <i>our unique Sapiens language</i>. Our language, alone of all the animals, enables us to talk about things that do not exist at all. You could never convince a monkey to give you a banana by promising him limitless bananas after death, in monkey heaven.</span>

> <span class='red'>By combining profound insights with a remarkably vivid language, Sapiens acquired cult status among diverse audiences, captivating teenagers as well as university professors, animal rights activists alongside government ministers.</span>


# Introduction to Recurrent Neural Networks

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' /> 
</p>


<span class='red'>I-know-everything:</span>  So, Padwan today we are going to study about language through the lens of Neural Networks. Let me get philosophical for a bit, and show how we are what today because we are able to communicate which each other the ideas, the ideas to push the human race forward. Language has been a critical cornerstone to the foundation of human mankind and will also play a critical role in human-computer world.*Don't send terminator vibes, transmitting [JARVIS](https://marvel-movies.fandom.com/wiki/J.A.R.V.I.S.) vibes....*  

<span class='green'>I-know-nothing:</span> Does this mean that it will be like Image where computer understand only numbers, the underlying language will be converted to numbers and where neural network does it's magic?

<span class='red'>I-know-everything:</span> Bingo, that's exactly right. As Image understanding has it's own challenges like occlusion, viewpoint variation, deformations, background clutter, etc., dealing with language comes with it's own challenges, starting from what language we are dealing with. This is what Natural Language Processing (NLP) field is about.  

Let me jump and tell you about the idea of what is <span class='purple'> Force of RNN </span>. RNN known as recurrent neural networks or Elman Network are useful for dealing with sequential information.

<span class='green'>I-know-nothing:</span> Why a new network, can't we just use the old <span class='purple'>[Force of MLP](https://dudeperf3ct.github.io/mlp/mnist/2018/10/08/Force-of-Multi-Layer-Perceptron/)</span> or <span class='purple'>[Force of CNN](https://dudeperf3ct.github.io/cnn/mnist/2018/10/17/Force-of-Convolutional-Neural-Networks/)</span>? for dealing with these sequential information? What's so special about sequential information? What makes RNN special to dealing with these types of data?

<span class='red'>I-know-everything:</span> Ahh, as we will see later, indeed we can use <span class='purple'>[Force of CNN](https://dudeperf3ct.github.io/cnn/mnist/2018/10/17/Force-of-Convolutional-Neural-Networks/)</span> for dealing with sequential data but <span class='purple'>[Force of MLP](https://dudeperf3ct.github.io/mlp/mnist/2018/10/08/Force-of-Multi-Layer-Perceptron/)</span> fails to capture relationships found in sequential data. There are different types of sequential data (where some sort of sequence is present) like time series, speech, text, financial data, video, etc. These are sequential data are not independent but rather sequentially correlated. For e.g. given a sequence, The cat sat on mat. Here, from given example we come to a conclusion that cat is sitting on mat. So, there is some context, which is nearby values of the data are related to each other. This is one example of sequential data and we can see how the context in such sequetial data matters to understand sequences. Same can be said for video, audio or any other sequential i.e. sequence involving data.

- What makes RNN special? 

This will take us on a journey to understand what are RNN. How they so effectively learn sequential data and what can we use them for?

<p align="center">
<img src='/images/rnn/simple_rnn.png' /> 
</p>

So, what can we infer by looking at the figure above. There is some context(t) which take in two inputs, Input(t) and context(t-1), which then produces output(t). Also, context(t-1) gets updated to context(t). There is some form of recursion. This is a Simple RNN which take in sequence input, to produce output. where context(t-1) is known as state. We will explore this in-detail.

There are different types of sequence input and output combination that can be used across various applications. 

<p align="center">
<img src='/images/rnn/applications.jpeg' /> 
</p>

In above figure, inputs are green, output blue and RNN's state in green. From left to right, One-to-One is what we saw in CNNs image classification where input image in, prediction output which class image belongs to i.e. fixed-size input to fixed-size output. One-to-Many contains sequence output, for fixed-input size, this can be task of image captioning where input is image and output is sentences of words. *We know how multiple characters make up a word and multiple words combine to make a sentence.* Many-to-one, here input is sequence and output is single prediction, which can be related to task of sentiment analysis, wherein input is sequence of words i.e. movie review and output is whether review is positive, neutral or negative. Next, Many-to-Many, here both input and output are sequence of words, which also happens in Machine Translation, where we input some sentence in English and get output sequence of words in French of varying length sequence. Another variant of Many-to-many, this can be related to video classification where we wish to label each frame in video.

We still haven't answered what makes them special. Let's deep dive and take apart RNN and assemble it to understand what makes RNNs special.

We have looked at how simple MLP works. We define, $$\mathbf{h}$$ = $$\phi(\mathbf{W}\mathbf{x})$$ , where $$\phi$$ is an activation function and $$\mathbf{y}$$ = $$\mathbf{V}\mathbf{h}$$, where $$\mathbf{V}$$ is weight matrix connecting hidden and output layers, $$\mathbf{W}$$ weight matrix connecting input and hidden layer and $$\mathbf{x}$$ is input vector. We also looked at different types of activation functions.

<p align="center">
<img src='/images/rnn/nn.png' /> 
</p>

When we look at sequences of video frames, we use only the images as input to CNN and completely ignore sequential aspects present in the video. Taking example from [Edwin Chen's blog](http://blog.echen.me/2017/05/30/exploring-lstms/), if we see a scence of beach, we should boost beach activities in future frames: an image of someone in the water should probably be labeled *swimming*, not *bathing*, and an image of someone lying with their eyes closed is probably *suntanning*. If we remember that Bob just arrived at a supermarket, then even without any distinctive supermarket features, an image of Bob holding a slab of bacon should probably be categorized as *shopping* instead of *cooking*.

We need to integrate some kind of state which keeps tracks the current view of world for the model by continually updating as it learns new things. It will function like internal memory.

After modifying the above equation to incorporate some notion that our model keeps remembering bits of information, new equation looks like,

$$
\begin{aligned}
\mathbf{h}_{t} & = \phi(\mathbf{W}\mathbf{x}_{t} + \mathbf{U}\mathbf{h}_{t-1}) \\
\mathbf{y}_{t} & = \mathbf{V}\mathbf{h}_{t}
\end{aligned}
$$

Here, $$\mathbf{h}_{t}$$, hidden layer of network acts as internal memory storing useful information about input and passing the same info to next hidden layer so that it can update the state (internal memory or hidden layer) as new input comes. In this way, hidden layer sort of contains all this history of past inputs.

<p align="center">
<img src='/images/rnn/rnn.png' /> 
</p>


This is where the recurrent word comes into RNN, as we are using the same state(hidden layer) for every input again and again. Another way to think about how RNN works is, we get an input, our hidden layer captures some information about that input, and then when next input comes, the information in hidden layer gets updated according to new input but also keeping some of the previous inputs. So in all, hidden layer becomes an internal memory which captures information about what has been calculated so far. The below diagram shows unrolled RNN, if sequence contains 3 words, then the network will be unrolled into 3-layer network as shown below.


<p align="center">
<img src='/images/rnn/unfold_rnn.png' /> 
</p>



<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology

Force of RNN - Recurrent Neural Networks

loss function - cost, error or objective function


---

# Further Reading

Must Read! [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

[Chater 9 Book: Speech and Language Processing by Jurafsky & Martin](https://web.stanford.edu/~jurafsky/slp3/9.pdf)



[Generating Text with Recurrent Neural Networks](https://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf)

[Recurrent neural network based language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)

[A neural probabilistic language model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

[A Primer on Neural Network Modelsfor Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf)

[Extensions of Recurrent neural network language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2011/mikolov_icassp2011_5528.pdf)

[Wildml Introduction to RNN](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)

[Machine Learning for Sequential Data: A Review](http://web.engr.oregonstate.edu/~tgd/publications/mlsd-ssspr.pdf)

---

# Footnotes and Credits


[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)

[Simple RNN](www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)

[Examples of RNN](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)


---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

