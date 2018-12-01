---
layout:     post
title:      Introduction to Multi-Layer Perceptron
date:       2018-10-10 12:00:00
summary:    This post will provide an hands-on-tutorial and insight into MLP using MNSIT dataset and Keras and PyTorch frameworks.
categories: neural-networks, mlp, keras, python, mnist
published : true
---

# MLP

In this post, we will go through basics of MLP using MNIST dataset. We will implement this using two popular deep learning frameworks `Keras` and `PyTorch`.

Hey yo, but what is MLP? what is MNIST? 

Well sit tight and buckle up. I will go through everything in-detail.


```python
# load all the required libraries

import numpy as np                                    # package for computing
from sklearn.model_selection import train_test_split  # split dataset
import keras                                          # import keras with tensorflow as backend
from keras.datasets import mnist                      # import mnist dataset from keras 
from keras.models import Model, Sequential            # sequential and functional api keras 
from keras.layers import Dense, Input                 # dense and input layer for constructing mlp

import matplotlib.pyplot as plt             # matplotlib library for plotting

# display plots inline (in notebook itself)

%matplotlib inline                          
```

    Using TensorFlow backend.


## MNIST Dataset


 
The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples each of size 28 x 28 pixels. The digits have been size-normalized and centered in a fixed-size image.

It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting. 

Here is one example from dataset





```python
# load mnist data

# the data, split between train and validation sets
(train_x, train_y), (test_x, test_y) = mnist.load_data()

#orginally shape (60000, 28, 28) for train and (10000, 28, 28) for test
#but as we will be using fully connected layers we will flatten
#the images into 1d array of 784 values instead of (28 x 28) 2d array
train_x = train_x.reshape(60000, 784)
test_x = test_x.reshape(10000, 784)

# As image is grayscale it has values from [0-255] which we will visualize below
# convert dtype to float32 and scale the data from [0-255] to [0-1]
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_x /= 255
test_x /= 255

print('Training samples and shape:', train_x.shape[0], train_x.shape)
print('Test samples and shape:', test_x.shape[0], test_x.shape)
```

    Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
    11493376/11490434 [==============================] - 0s 0us/step
    Training samples and shape: 60000 (60000, 784)
    Test samples and shape: 10000 (10000, 784)


## Different sets of splitting data

Wait hold on second, what are these different sets?


We usually define 

 - Training set - which you run your model (or learning) algorithm on.
 - Dev (development) or val (validation) set - which you use to tune parameters, select features, and make other decisions regarding learning algorithms or model. Sometimes also called out as hold-out cross validation set 
 - Test set - which you use to evaluate the performance of algorithm, but not to make any decisions regarding what the model or learning algorithm or parameters to use.
 
The `dev` and `test` set allow us to quickly see how well our model is doing.
 
### Cat Classifier 
 
Consider a scenario where we are building cat classifier (cats really, why not!). We run a mobile app, and users are
uploading pictures of many different things to the app. 

We collect a large training set by downloading pictures of cats (positive examples) and non-cats (negative examples) off of different websites. We split the dataset 70% / 30% into training and test sets. Using this data, we build a cat detector that works well on the
training and test sets. But when we deploy this classifier into the mobile app, we find that the performance is
really poor!

What happened?

Since training/test sets were made of website images, our algorithm did not generalize well to the actual distribution you care about: mobile phone pictures.

Before the modern era of big data, it was a common rule in machine learning to use a random 70% / 30% split to form  training and test sets. This practice can work, but it’s a bad idea in more and more applications where the training distribution (website images in our example above) is different from the distribution you ultimately care about (mobile
phone images).

**Lesson:** `Choose dev and test sets to reflect data you expect to get in the future and want to do well on.`




```python
# we will split val into --> 20% val set and 80% test set 
# stratify ensures the distribution of classes is same in both the sets

val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=0.8, stratify=test_y)

print ('Validation samples and shape', val_x.shape[0], val_x.shape)
print ('Test samples and shape', test_x.shape[0], test_x.shape)
```

    Validation samples and shape 2000 (2000, 784)
    Test samples and shape 8000 (8000, 784)


## Visualization of data

Enough talk, show me the data!


```python
# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(train_x[idx].reshape(28, 28), cmap='gray')
    # print out the correct label for each image
    ax.set_title(str(train_y[idx]))
```


![Sample Images from dataset](/images/mnist_mlp_files/mnist_mlp_7_0.png)



```python
img = train_x[1].reshape(28, 28)

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')
```


![Example 0](/images/mnist_mlp_files/mnist_mlp_8_0.png)


# Introduction to MLP



MLP is multi-layer percepton. Perceptron is a single layer neural network and a multi-layer perceptron is called Neural Networks.  
We have seen the dataset, which consist of [0-9] numbers and images of size 28 x 28 pixels of values in range [0-1] . 

Now, <font color='green'>Mr.I-know-nothing</font> being too lazy to find which number is what asks for <font color='red'>Mr.I-know-everything</font> apprenticeship to create a Machine Learning Model such that if we pass a grayscale image of size 28 x 28 pixels to the model, it outputs a correct label corresponding to that image. 

<font color='blue'> A long time ago in a galaxy far, far away.... </font> </br>

<font color='green'>Mr.I-know-nothing:</font> Master, how can I create such a intelligent machine to recognize and label given images?

<font color='red'>Mr.I-know-everything:</font> Young Padwan, we will use the `Force of Neural Networks` inspired from our brain. Here, let me take you on a journey of one example for example 0. We have 784 pixel values in range  [0-1] describing what zero looks like (pixels bright in the center in shape of 0 and dark like the dark side elsewhere). 0 passes through the network like the one shown below and return 10 values which will help in classfying the image is 0 or 1 or 2 and so on. 

<font color='green'>Mr.I-know-nothing:</font> How will the number decide which image is what label?

<font color='red'>Mr.I-know-everything:</font> If the image passed is 0 (also known as `forward pass`), the network will output array [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]. The first place 1 indicates the image passed is 0.

<font color='green'>Mr.I-know-nothing:</font> How does the network learn such a magic trick?

<font color='red'>Mr.I-know-everything:</font> Young Padwan, you are learning to ask right questions. I will give 2 explainations so listen closely. First let me give you an intutive explaination. The neural networks train themselves  repetitively on data so that they can adjust the weights in each layer of the network to get the final result closer to given label. Now the second explaination in jargon words, as shown in the network we have input layer, hidden layer and output layer. Okay? So, input layer has 784 nodes (neurons) i.e. it accepts 784 values which is exactly our example 0 has. Next node is hidden layer which contains 16 neuron and what are its values? They are randomly initialized. Next is the output layer which has 10 nodes. These are the values which our network gives us after performing special operations which we will then compare to our desired label which is zero in this case.

<font color='green'>Mr.I-know-nothing:</font> What if network outputs does not match our desired result?

<font color='red'>Mr.I-know-everything:</font> That means, our network is stupid (for now). But it learns, it learns from its mistakes. The process by which it learns is backpropogation. So, in `jar jar backpropogation`, in our example desired result was [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] and network outputs [0.24, 0.542, 0.121, 0.32, 0.56, 0.67, 0.213, 0.45, 0.312, 0.98] which in this case is 9 (highest value). So, now network tells its previous layer (also known as `backward pass`), hidden layer hey look you gave me wrong answer 9, see here the right answer was 0 which is called as `loss`. Make necessary changes with help of chain rule to your weights so that when next time you see 0, you will improve the prediction in such a way that output will be also 0.

<font color='green'>Mr.I-know-nothing:</font> Does repeating these telling the correct results and correcting the wrong results is what `Force of Neural Networks` all about?

<font color='red'>Mr.I-know-everything:</font> Well, if you put it that way, you are sucking all the fun out of magic. But yes, this is what is called `supervised learning`, where network is supervised to show it direction so that it does not get lost in the woods ([Out of Woods](https://www.youtube.com/watch?v=JLf9q36UsBk)).

<font color='green'>Mr.I-know-nothing:</font> This is all intutive understanding with some jargon words. What about real equations? I mean, everywhere I see there are equations. Where are they?

<font color='red'>Mr.I-know-everything:</font> They are bit scary but if you insist I will write them for you.


$\mathbf{x} : \textrm{Input layer with 784 values} (\mathbf{x_1}, \mathbf{x_2},..., \mathbf{x_{784}}) \\
\mathbf{w\rm\normalsize i} : \textrm {Weights of hidden layer with 16 values} (\mathbf{w\rm\normalsize i_1}, \mathbf{w\rm\normalsize i_2},....., \mathbf{w\rm\normalsize i_{16}})\\ 
\mathbf{w\rm\normalsize i_0} : \textrm {bias of hidden layer} \\
\mathbf{w\rm\normalsize o} : \textrm {Weights of output layer with 10 values} (\mathbf{w\rm\normalsize o_1}, \mathbf{w\rm\normalsize o_2},....., \mathbf{w\rm\normalsize o_{10}})\\ 
\mathbf{w\rm\normalsize o_0} : \textrm {bias of output layer} \\ 
\mathbf{\hat{y}} : \textrm {prediction} \\ 
\mathbf{y} : \textrm{desired result} \\ $


**Forward Pass**

\begin{aligned}
\mathbf{a} = \mathbf{w\rm\normalsize i^T} \mathbf{x} + \mathbf{w\rm\normalsize i_0} \\
\mathbf{h} = f(\mathbf{a}) = f(\mathbf{w\rm\normalsize i^T} \mathbf{x} + \mathbf{w\rm\normalsize i_0}) \\
\mathbf{z} = \mathbf{w\rm\normalsize o^T} \mathbf{h} + \mathbf{w\rm\normalsize o_0} \\
\mathbf{\hat{y}} = softmax(\mathbf{z}) \\
\\~\\
\textrm{Relu  Non-linearity}: f(\mathbf{k}) = max(k, 0) \\
\textrm{Softmax  Function}: \sigma_j(\mathbf{z})  = \frac {\exp(\mathbf{z}_j)} {\sum_{k}^{nclass} \exp(\mathbf{z}_k)} \\
\end{aligned}

**Error Function**

\begin{aligned}
E = l(\mathbf{y}, \mathbf{\hat{y}}) = -\sum_{i}^{nclass}\mathbf{y_i}ln{\mathbf{\hat{y_i}}}
\end{aligned}

**Backward Pass**

\begin{aligned}
\frac{\partial E}{\partial \mathbf{\hat{y_i}}} = - \frac {\mathbf{y_i}}{\mathbf{\hat{y_i}}} \\
\frac{\partial \mathbf{\hat{y_i}}}{\partial \mathbf{z}} = 
\begin{cases}
\frac {\exp(\mathbf{z}_i)} {\sum_{k}^{nclass} \exp(\mathbf{z}_k)} - (\frac {\exp(\mathbf{z}_i)} {\sum_{k}^{nclass} \exp(\mathbf{z}_k)})^2   &i=k    \\
(\frac {e^{(\mathbf{z}_i)}e^{(\mathbf{z}_k)}} {\sum_{k}^{nclass} \exp(\mathbf{z}_k)})^2 &i \ne k    \\
\end{cases} \\
&=
\begin{cases}
\mathbf{\hat{y_i}}(1-\mathbf{\hat{y_i}})  &i=k    \\
-\mathbf{\hat{y_i}}\mathbf{\hat{y_k}} &i \ne k    \\
\end{cases} \\
\frac{\partial E}{\partial \mathbf{z_i}} = \sum_{k}^{class}\frac{\partial E}{\partial \mathbf{\hat{y_k}}}\frac{\partial \mathbf{\hat{y_k}}}{\partial \mathbf{z_i}} \\
& = \frac{\partial E}{\partial \mathbf{\hat{y_i}}}\frac{\partial \mathbf{\hat{y_i}}}{\partial \mathbf{z_i}} - \sum_{i \ne k}\frac{\partial E}{\partial \mathbf{\hat{y_k}}}\frac{\partial \mathbf{\hat{y_k}}}{\partial \mathbf{z_i}} \\
& = \sum_{k}^{class}\frac{\partial E}{\partial \mathbf{\hat{y_i}}}\frac{\partial \mathbf{\hat{y_i}}}{\partial \mathbf{z_i}} \\
& = -\mathbf{\hat{y_i}}(1-\mathbf{y_i}) + \sum_{k \ne i}\mathbf{\hat{y_k}}\mathbf{y_i} \\
& = -\mathbf{\hat{y_i}} + \mathbf{y_i}\sum_{k}\mathbf{\hat{y_k}} \\
& = \mathbf{\hat{y_i}} - \mathbf{y_i} \\
\frac{\partial E}{\partial \mathbf{w\rm\normalsize o_{ji}}} = \sum_{i}\frac{\partial E}{\partial \mathbf{z_i}}\frac{\partial \mathbf{z_i}}{\partial \mathbf{w\rm\normalsize o_{ji}}} \\
& = (\mathbf{\hat{y_i}} - \mathbf{y_i})\mathbf{h_j} \\
\frac{\partial E}{\partial \mathbf{w\rm\normalsize o_{0}}}=\frac{\partial E}{\partial \mathbf{z_i}} = \mathbf{\hat{y_i}} - \mathbf{y_i} \\
\frac{\partial E}{\partial \mathbf{h_{ji}}} = \sum_{i}\frac{\partial E}{\partial \mathbf{z_i}}\frac{\partial \mathbf{z_i}}{\partial \mathbf{h_{ji}}} \\
& = (\mathbf{\hat{y_i}} - \mathbf{y_i})\mathbf{w\rm\normalsize o_j} \\
\frac{\partial \mathbf{h_{ji}}}{\partial \mathbf{a}} =
\begin{cases}
1   &a>0    \\
0 &else    \\
\end{cases} \\
\frac{\partial E}{\partial \mathbf{w\rm\normalsize i_{ji}}} = \sum_{i}\frac{\partial E}{\partial \mathbf{h_{ji}}}\frac{\partial \mathbf{h_{ji}}}{\partial \mathbf{a}}\frac{\partial \mathbf{a}}{\partial \mathbf{w\rm\normalsize i_{ji}}} \\
& = (\mathbf{\hat{y_i}} - \mathbf{y_i})\mathbf{w\rm\normalsize o_j}\mathbf{x\rm\normalsize _j} \\
\frac{\partial E}{\partial \mathbf{w\rm\normalsize i_{0}}}=\frac{\partial E}{\partial \mathbf{h_{ji}}} = (\mathbf{\hat{y_i}} - \mathbf{y_i})\mathbf{w\rm\normalsize o_j} \\
\end{aligned}

<font color='red'>Mr.I-know-everything:</font> I am sure you got lot of questions now. So, shoot.
<font color='green'>Mr.I-know-nothing:</font> Wow! That's mouthful! What is $\mathbf{w\rm\normalsize i_{0}}$ and $\mathbf{w\rm\normalsize o_{0}}$ ? What is the function $f(\mathbf{h})$ ? What are we doing in backpropogation? Is backpropogation only the way to propogate calculate error?

<font color='red'>Mr.I-know-everything:</font> Wooh slow down! Okay let me answer one by one.

1. What is $\mathbf{w\rm\normalsize i_{0}}$ and $\mathbf{w\rm\normalsize o_{0}}$ ?

These are called biases. A layer in a neural network without a bias is nothing more than the multiplication of an input vector with a matrix. Using a bias, you’re effectively adding another dimension to your input space.

2. What is the function $f(\mathbf{h})$ ?

This functon plays an important role in machine learning. This types function are called non-linear functions. By introducing them in our network we introduce non-linearlity, non-linear means that the output cannot be reproduced from a linear combination of the inputs. Another way to think of it is if we don't use a non-linear activation function in the network, no matter how many layers it had, the network would behave just like a single-layer perceptron, because summing these layers would give you just another linear function and most of the problems in real world are non-linear. Non-linearity is needed in activation functions because its aim in a neural network is to produce a nonlinear decision boundary via non-linear combinations of the weight and inputs. To provide a better seperation for higher dimensional data then a simple line seperator using linear function.There are several types of non-linear functions.

   a. Relu Function

   b. Tanh Function

   c. Sigmoid Function

   d. Leaky Relu

   e. ELU

   f. PRelu and [many more](https://en.wikipedia.org/wiki/Activation_function).
   
   
3. What are we doing in backprop and is it the only way?

While designing a Neural Network, in the beginning, we initialize weights with some random values or any variable for that fact. So, it’s not necessary that whatever weight values we have selected will be correct, or it fits our model the best. Okay, fine, we have selected some weight values in the beginning, but our model output is way different than our actual output i.e. the error value is huge.

Now, how will you reduce the error?
Basically, what we need to do, we need to somehow explain the model to change the parameters (weights), such that error becomes minimum.That means, we need to train our model. One way to train our model is through Backpropagation but it is not the only way. There is another method called Synthetic Gradient designed by the Jedi Council. We will visit them later. If you are curious, look them up [here](https://iamtrask.github.io/2017/03/21/synthetic-gradients/) and [here](https://www.youtube.com/watch?v=1z_Gv98-mkQ)

In short, backprop algorithm looks for the minimum value of the error function in weight space using a technique called gradient descent. The weights that minimize the error function is then considered to be a solution to the learning problem. 

Gradient Descent is like descending a mountain blind folded. And goal is to come down from the mountain to the flat land without assistance. The only assistance you have is a gadget which tells you the height from sea-level. What would be your approach be. You would start to descend in some random direction and then ask the gadget what is the height now. If the gadget tells you that height and it is more than the initial height then you know you started in wrong direction. You change the direction and repeat the process. This way in many iterations finally you successfully descend down.

This is what gradient descent does. It tells the model which direction to move to minimize the error. There are different optimizer which tell us how can we find this direction. 

   a. Vanilla Gradient
   
   b. Adam
   
   c. RMS Prop
   
   d. SGD
   
   e. Nestrov Momentum


And this is behind the scenes (BTS) of how a `Force of Neural Network` learns.

<span color='green'>Mr.I-know-nothing:</span> Thank you Master, I follow.

<span color='red'>Mr.I-know-everything:</span> Now you are in for a treat. As you have learn about what different terms and functions are used to train a neural network. We will dive-in implementation using `Keras saber`. Here backpropogation is already implemented i.e. you only need to design forward pass and loss(or error) function, the framework takes care to backward pass. 

## Sequential API


```python
# [0-9] unique labels
num_classes = 10
epochs = 5
batch_size = 32
```


```python
# convert class vectors to binary class matrices
train_y = keras.utils.to_categorical(train_y, num_classes)
val_y = keras.utils.to_categorical(val_y, num_classes)
test_y = keras.utils.to_categorical(test_y, num_classes)
print ('Training labels shape:', train_y.shape)
```

    Training labels shape: (60000, 10)



```python
model = Sequential()
model.add(Dense(784, activation='relu', input_shape=(784,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 784)               615440    
    _________________________________________________________________
    dense_2 (Dense)              (None, 16)                12560     
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                170       
    =================================================================
    Total params: 628,170
    Trainable params: 628,170
    Non-trainable params: 0
    _________________________________________________________________



```python
history = model.fit(train_x, train_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(val_x, val_y))
```

    Train on 60000 samples, validate on 2000 samples
    Epoch 1/5
    60000/60000 [==============================] - 45s 754us/step - loss: 0.2280 - acc: 0.9324 - val_loss: 0.1070 - val_acc: 0.9675
    Epoch 2/5
    60000/60000 [==============================] - 45s 754us/step - loss: 0.0858 - acc: 0.9744 - val_loss: 0.0932 - val_acc: 0.9690
    Epoch 3/5
    60000/60000 [==============================] - 45s 751us/step - loss: 0.0574 - acc: 0.9823 - val_loss: 0.0877 - val_acc: 0.9730
    Epoch 4/5
    60000/60000 [==============================] - 48s 804us/step - loss: 0.0413 - acc: 0.9866 - val_loss: 0.0606 - val_acc: 0.9815
    Epoch 5/5
    60000/60000 [==============================] - 50s 834us/step - loss: 0.0299 - acc: 0.9903 - val_loss: 0.0791 - val_acc: 0.9765



```python
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
```


![Accuracy plots](/images/mnist_mlp_files/mnist_mlp_16_0.png)



![Loss plots](/images/mnist_mlp_files/mnist_mlp_16_1.png)



```python
score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    Test loss: 0.0735809408277
    Test accuracy: 0.97975



```python
# obtain one batch of test images
images, labels = test_x[:32], test_y[:32]

# get sample outputs
predict = model.predict_on_batch(images)
# convert output probabilities to predicted class
preds = np.argmax(predict, axis=1)
labels = np.argmax(labels, axis=1)

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(images[idx].reshape((28, 28)), cmap='gray')
    ax.set_title("{} ({})".format(str(preds[idx]), str(labels[idx])),
                 color=("green" if preds[idx]==labels[idx] else "red"))
```


![Test results](/images/mnist_mlp_files/mnist_mlp_18_0.png)


## Functional API


```python
# [0-9] unique labels
num_classes = 10
epochs = 5
batch_size = 32
```


```python
inputs = Input(shape=(784,))
x = Dense(784, activation='relu')(inputs)
x = Dense(16, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
```


```python
history = model.fit(train_x, train_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(val_x, val_y))
```

    Train on 60000 samples, validate on 2000 samples
    Epoch 1/5
    60000/60000 [==============================] - 49s 816us/step - loss: 0.2154 - acc: 0.9365 - val_loss: 0.1260 - val_acc: 0.9590
    Epoch 2/5
    60000/60000 [==============================] - 48s 801us/step - loss: 0.0833 - acc: 0.9740 - val_loss: 0.0766 - val_acc: 0.9730
    Epoch 3/5
    60000/60000 [==============================] - 48s 794us/step - loss: 0.0570 - acc: 0.9821 - val_loss: 0.0793 - val_acc: 0.9755
    Epoch 4/5
    60000/60000 [==============================] - 45s 757us/step - loss: 0.0408 - acc: 0.9869 - val_loss: 0.0724 - val_acc: 0.9780
    Epoch 5/5
    60000/60000 [==============================] - 44s 740us/step - loss: 0.0321 - acc: 0.9896 - val_loss: 0.1023 - val_acc: 0.9725



```python
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
```


![Accuracy plot](/images/mnist_mlp_files/mnist_mlp_23_0.png "Accuracy plot")



![Loss plot](/images/mnist_mlp_files/mnist_mlp_23_1.png "Loss plot")



```python
score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```


```python
# obtain one batch of test images
images, labels = test_x[:32], test_y[:32]

# get sample outputs
predict = model.predict_on_batch(images)
# convert output probabilities to predicted class
preds = np.argmax(predict, axis=1)
labels = np.argmax(labels, axis=1)

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(images[idx].reshape((28, 28)), cmap='gray')
    ax.set_title("{} ({})".format(str(preds[idx]), str(labels[idx])),
                 color=("green" if preds[idx]==labels[idx] else "red"))
```

<span color='red'>Mr.I-know-everything:</span> Young Padwan, now you have the same power as me to train an MLP. Now knock yourself and experiement with different number of layers. Also, watch for training and validation loss as hint if model is moving in right direction. There you will come across `overfitting` and `underfiting`. So, be sure to watch them and we will discuss about them in detail in next time where you will learn about `Force of CNN` and how they can further give us best model (Yes, better than MLP). Until next time, try different architectures and keep researching.

<span color='green'>Mr.I-know-nothing:</span> Thank you Master.

Happy Learning! :tada:
