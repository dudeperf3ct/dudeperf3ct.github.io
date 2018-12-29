---
layout:     post
title:      Power of Transfer Learning
date:       2018-12-03 12:00:00
summary:    This post will provide an brief introduction to Transfer Learning using Dogs vs Cats Redux Competition dataset from Kaggle along with the implementation in Keras framework.
categories: transfer learning catsvsdogs
published : true
---


# Transfer Learning

In this notebook, we will go through basics of Transfer Learning using [Cats vs Dogs Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) Competition dataset from kaggle. We will implement this using one of the popular deep learning framework <span class='yellow'>Keras</span> . 

> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/CNN/mnist_cnn_keras.ipynb), [PyTorch](https://github.com/dudeperf3ct/DL_notebooks/blob/master/CNN/mnist_cnn_pytorch.ipynb), [Tensorflow](https://github.com/dudeperf3ct/DL_notebooks/blob/master/CNN/mnist_cnn_tensorflow.ipynb) and [fastai](https://github.com/dudeperf3ct/DL_notebooks/blob/master/CNN/mnist_cnn_fastai.ipynb).  

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but what is Transfer Learning?

Well sit tight and buckle up. I will go through everything in-detail.

-insert transfer learning meme-

Feel free to jump anywhere,
- [Learning Curves](#learning-curves)
- [Introduction to Transfer Learning](#introduction-to-transfer-learning)
  - [Transfer Learning scenarios](#2-major-transfer-learning-scenarios)
  - [When and how to finetune?](#next-question-would-be-when-and-how-to-fine-tune?)
- [Recap](#recap)
- [Keras](#keras)
  - [ConvNet as feature extractor](#convNet-as-feature-extractor)
  - [Finetuning](#fine-tuning)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)

# Dogs vs Cats Redux dataset

The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id. For each image in the test set, the task is to predict a probability that the image is a dog (1 = dog, 0 = cat).

<p align="center">
<img src='/images/transfer_learning_files/cats_dogs.jpg' />
</p>


# Data Preprocessing

Getting a data from kaggle using Kaggle API is a little tricky part, once done whole road is clear to play with data. For brevity, I will leave that part in notebooks and suppose that all data is downloaded.

The dataset once unzipped we get the data in following directory structure.

data/<br>
&nbsp;&nbsp;    train/<br>
&nbsp;&nbsp;&nbsp;&nbsp;        dog001.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;        dog002.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;        ...<br>
&nbsp;&nbsp;&nbsp;&nbsp;        cat001.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;        cat002.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;        ...<br>
&nbsp;&nbsp;    test/<br>
 &nbsp;&nbsp;&nbsp;&nbsp;       001.jpg<br>
 &nbsp;&nbsp;&nbsp;&nbsp;       002.jpg<br>
 &nbsp;&nbsp;&nbsp;&nbsp;       ...<br>


We will convert the above directory structure into this structure for ease of data processing.

data/<br>
&nbsp;&nbsp;    train/<br>
&nbsp;&nbsp;&nbsp;&nbsp;         dogs/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;             dog001.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;             dog002.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;             ...<br>
&nbsp;&nbsp;&nbsp;&nbsp;         cats/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;             cat001.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;             cat002.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;             ...<br>
&nbsp;&nbsp;    test/<br>
&nbsp;&nbsp;&nbsp;&nbsp;         001.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;         002.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;         003.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;         ...<br>


```python
print ('Training set images', len(os.listdir(train_cats_dir))+len(os.listdir(train_dogs_dir)))
print ('Validation set images', len(os.listdir(val_cats_dir))+len(os.listdir(val_dogs_dir)))
print ('Test set images', len(os.listdir(test_path)))
```

    Training set images 20000
    Validation set images 5000
    Test set images 12500


## Visualization of data

Enough talk, show me the cats and dogs!


```python
def preprocess_img(img, ax, label, train_dir):
    im = Image.open(os.path.join(train_dir, img))
    size = im.size
    ax.imshow(im)
    ax.set_title(f'{label} {size}')
```


```python
train_x = os.listdir(train_cats_dir)
# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 10))

for idx in np.arange(10):
    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
    preprocess_img(train_x[idx], ax, 'cat', train_cats_dir)
    # print out the correct label for each image
```


<p align="center">
<img src='/images/transfer_learning_files/transfer_learning_keras_26_0.png' />
</p>


```python
train_x = os.listdir(train_dogs_dir)
# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 10))

for idx in np.arange(10):
    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
    preprocess_img(train_x[idx], ax, 'dog', train_dogs_dir)
    # print out the correct label for each image
```


<p align="center">
<img src='/images/transfer_learning_files/transfer_learning_keras_27_0.png' />
</p>


# Learning curves

Lets dive into interpreting learning different curves to understand and ways to avoid underfitting-overfitting or bias-variance tradeoff. There is some sort of tug of war between bias and variance, if we reduce bias error that leads to increase in variance error.

Let's recap what we had from our previous discussion on bias and variance.

- (Training - Dev) error high ==> High variance ==> Overfitting ==> Add more data to training set
- Training error high     ==> High bias    ===> Underfitting ==> Make model more complex
- Bayes error ==> Optimal Rate  ==> Unavoidable bias
- (Training - Bayes) error ===> Avoidable bias
- Bias = Optimal error rate (“unavoidable bias”) + Avoidable bias


## Cat Classifier

Cats again! Suppose we run the algorithm using different training set sizes. For example, if you have 1,000 examples, we train separate copies of the algorithm on 100, 200, 300, ..., 1000 examples. Following are the different learning curves, where desired performance(green) along with dev(red) error and train(blue) error are plotted against the number of training examples.

Consider this learning curve,

<p align="center">
<img src='/images/transfer_learning_files/high_bias_or_variance.png' />
</p>


Is this plot indicating, high bias, high variance or both?

The training error is very close to desired performance, indicating avoidable bias is very low. The training(blue) error curve is relatively low, and dev(red) error is much higher than training error. Thus, the bias is small, but variance is large. As from recap above, adding more training data will help close gap between training and dev error and help reduce high variance.

Consider this curve,

<p align="center">
<img src='/images/transfer_learning_files/significant_variance_bias.png' />
</p>


Is this plot indicating, high bias, high variance or both?

This time, training error is large, as it is much higher than desired performance. There is significant avoidable bias. The dev error is also much larger than training error. This indicated we have significant bias and significant variance in our plot. We will use the ways to avoid both variance and bias.


Consider this curve,

<p align="center">
<img src='/images/transfer_learning_files/high_avoidable_bias.png' />
</p>

Is this plot indicating, high bias, high variance or both?

The training error is much higher than desired performance. This indicates it has high avoidable bias. The gap between training and dev error curves is small, indicating small variance.


**Lessons:**

- As we add more training data, training error can only get worse. Thus, the blue training error curve can only stay the same or go higher, and thus it can only get further away from the (green line) level of desired performance.

- The red dev error curve is usually higher than the blue training error. Thus, there’s almost no way that adding more data would allow the red dev error curve to drop down to the desired level of performance when even the training error is higher than the desired level of performance.


#### Techniques to reduce avoidable bias

1. Increase model size (number of neurons/layers)

This technique reduces bias by fitting training set better. If variance increases, we can use regularization to minimize the effect of increase in variance.

2. Modify input features based on insights from error analysis

Create additional features that help the algorithm eliminate a particular category of errors.These new features could help with both bias and variance.

3. Reduce or eliminate regularization (L2 regularization, L1 regularization, dropout)

This will reduce avoidable bias, but increase variance.

4. Modify model architecture (such as neural network architecture)

This technique can affect both bias and variance.

#### Techniques to reduce variance

1. Add more training data

This is the simplest and most reliable way to address variance, so long as you have access to significantly more data and enough computational power to process the data.

2. Add regularization (L2 regularization, L1 regularization, dropout)

This technique reduces variance but increases bias.

3. Add early stopping (i.e., stop gradient descent early based on dev set error)

This technique reduces variance but increases bias.

4. Feature selection to decrease number/type of input features

This technique might help with variance problems, but it might also increase bias. Reducing the number of features slightly (say going from 1,000 features to 900) is unlikely to have a huge effect on bias. Reducing it significantly (say going from 1,000 features to 100—a 10x reduction) is more likely to have a significant effect, so long as you are not excluding too many useful features. In modern deep learning, when data is plentiful, there has been a shift away from feature selection, and we are now more likely to give all the features we have to the algorithm and let the algorithm sort out which ones to use based on the data. But when your training set is small, feature selection can be very useful.

5. Modify model architecture and modify input features

These techniques are also mentioned in avoidable bias.

### Data Mismatch

Suppose a ML in a setting where the training and the dev/test distributions are different. Say, the training set contains Internet images + Mobile images, and the dev/test sets contain only Mobile images.

If the model generalizes well to new data drawn from the same distribution as the training set, but not to data drawn from the dev/test set distribution. We call this problem <span class='orange'>data mismatch</span> , since it is because the training set data is a poor match for the dev/test set data.

This illustration explains clearly the data mismatch problem.

<p align="center">
<img src='/images/transfer_learning_files/data_mismatch.png' />
</p>

- Training set

This is the data that the algorithm will learn from (e.g., Internet images + Mobile images). This does not have to be drawn from the same distribution as what we really care about (the dev/test set distribution).

- Training dev set

This data is drawn from the same distribution as the training set (e.g.,Internet images + Mobile images). This is usually smaller than the training set; it only needs to be large enough to evaluate and track the progress of our learning algorithm.

- Dev set 

This is drawn from the same distribution as the test set, and it reflects the distribution of data that we ultimately care about doing well on. (E.g., mobile images.)

- Test set

This is drawn from the same distribution as the dev set. (E.g., mobile images.)

### Techniques to resolve data mismatch problem

1. Try to understand what properties of the data differ between the training and the dev set distributions.

2. Try to find more training data that better matches the dev set examples that your algorithm has trouble with.

In next post, we will discuss about various regularization techniques and when and how to use them. Stay tuned!

# Introduction to Transfer Learning

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' />
</p>

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<span class='red'>I-know-everything:</span> Yo, my fellow apperentice. This time you will experience the <span class="purple"> Power of Transfer Learning</span>. Transfer Learning is a technique where you take a pretrained model trained on large dataset and transfer the learned knowledge to another model with small dataset but some what similar to large dataset for classification. For e.g. if we consider Imagenet dataset which contains 1.2 million images and 1000 categories, in that there are 24 different categories of dogs and 16 different categories of cats. So, we can transfer the learned features of cats and dogs from model trained on Imagenet dataset to our new model which contains 25,000 images of dogs and cats in training set.


<p align="center">
<img src='/images/transfer_learning_files/traditional_ml_setup.png' width="60%"/>
</p>


<p align="center">
<img src='/images/transfer_learning_files/transfer_learning_1.png' width="60%"/>
</p>

<span class='green'>I-know-nothing:</span> Ahh?

<span class='red'>I-know-everything:</span> Okay, let's take a step back and go over our learning from <span class='purple'>Force of CNN</span>. First, we saw what a convolution operator is, how different kernels or the numbers i n matrix give differnet results when applied to an image such as edge detector, blurring, sharpening, etc. After that, we visited different functions and looked at their properties and role in CNN, e.g. kernel, pooling, strides. We saw CNN consists of multiple CONV-RELU-POOL layers, followed by FC layers like the one shown below.

<p align="center">
<img src='/images/transfer_learning_files/tesla_cnn.png' width="70%"/>
</p>

We saw how the training a CNN is similar to MLP. It consists of forward pass followed by backward pass where the kernels adjust the weights so as to backpropogate the error in classification and also looked at different architectures and role they played in Imagenet competition. The only thing we did not discuss is that what these CNN are learning that makes them able to classify 1.2 million images in 1000 categories with 2.25% top5 error rate better than humans. <span class='saddlebrown'>What is going on insides these layers to them such better classifiers?</span>

<p align="center">
<img src='/images/transfer_learning_files/visualize_cnn.png' width="60%"/>
</p>

Many details of how these models works is still a mystery (black-box), but Zeiler and Fergus showed in their excellent [paper](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf) on Visualizaing and Understanding Convolution Neural Networks, that lower convolutional layers capture low-level image features, e.g. edges, while higher convolutional layers capture more and more complex details, such as body parts, faces, and other compositional features.

The final fully-connected layers are generally assumed to capture information that is relevant for solving the respective task, e.g. AlexNet's fully-connected layers would indicate which features are relevant to classify an image into one of 1000 object categories.

<p align="center">
<img src='/images/transfer_learning_files/layer1_layer2.png' />
</p>

<p align="center">
<img src='/images/transfer_learning_files/layer3.png' />
</p>

<p align="center">
<img src='/images/transfer_learning_files/layer4_layer5.png' />
</p>


As we observe in above pictures, different layers correspond or activate to different features in the images. For e.g., Layer 3 activates for different textures, Layer 2 activates for different edges and circles, similarly, Layer 5 activates for faces of humans, animals also, they learn to identify text in the image on their own.

In short, here is how CNN learns.

<p align="center">
<img src='/images/transfer_learning_files/layers_cnn.jpg' />
</p>

When an image of face of human is passed through CNN, the initial layers learn to identify simple features like nose, eyes, ears, etc. As we move up the architecture, the higher layers will combine simple features into more complex feature and finally dense layers at the top of the network will combine very high level features and produce classification predictions.

<span class='green'>I-know-nothing:</span> Now I understand what goes behind the scenes of CNN model. So, how can these features help us in training our model?

<span class='red'>I-know-everything:</span> Glad you asked. Transfer learning is an optimization, a shortcut to saving time or getting better performance. There are tthree possible benefits to look for when using transfer learning:

1. Higher start. The initial skill (before refining the model) on the source model is higher than it otherwise would be.

2. Higher slope. The rate of improvement of skill during training of the source model is steeper than it otherwise would be.

3. Higher asymptote. The converged skill of the trained model is better than it otherwise would be.


<p align="center">
<img src='/images/transfer_learning_files/Three-ways-in-which-transfer-might-improve-learning.png' width="60%"/>
</p>

On some problems where you may not have very much data, transfer learning can enable you to develop skillful models that you simply could not develop in the absence of transfer learning.

For a new classification task, we can simply use the off-the-shelf features of a state-of-the-art CNN pre-trained on ImageNet and train a new model on these extracted features.

Taking the example of our dataset, where pretrained model has already seen bunch of cats and dogs from Imagenet dataset and now the model is somewhat adept to know difference between cats and dogs and many other different things from Imagenet dataset. So, if we just train the pretrain model with our new dataset of just cats and dogs, the model has already learned how different types of dogs and cats looks like. Building on these knowledge of pretrained model, new model learns to classify the new dataset very quickly.

### 2 Major Transfer Learning scenarios

There are 2 major Transfer Learning scenarios:

1. ConvNet as fixed feature extractor

Take a ConvNet pretrained on ImageNet, remove the last fully-connected layer (this layer’s outputs are the 1000 class scores for a different task like ImageNet), then treat the rest of the ConvNet as a fixed feature extractor for the new dataset. For eg, in AlexNet, this would compute a 4096-D vector for every image that contains the activations of the hidden layer immediately before the classifier. We call these features <span class='orange'>CNN codes</span>. Once you extract the 4096-D codes for all images, train a linear classifier (e.g. Linear SVM or Softmax classifier) for the new dataset.

In our case of dogs and cats dataset, we will leverage any of the pretrained architectures mentioned in our previous post on CNN, and use them as base model for transfer learning. We will remove the final layer and insert a new hidden layers with desired number of classes for classification as per our new dataset. (or, also we could add multiple different number if hidden layers in between this pretrained model and our output classes) This is one type of transfer learning we can leverage without having to design new architecture and having to worry if it will even work for such smaller dataset(our cats and dogs dataset). This technique surprisingly gives amazing results in such less time and less compute power than the model trained from scratch will will takes hours and days of compute power to achieve similar or better results.

2. Finetuning the ConvNet

The second strategy is to not only replace and retrain the classifier on top of the ConvNet on the new dataset, but to also fine-tune the weights of the pretrained network by continuing the backpropagation. It is possible to fine-tune all the layers of the ConvNet, or it’s possible to keep some of the earlier layers fixed (due to overfitting concerns) and only fine-tune some higher-level portion of the network. This is motivated by the observation that the earlier features of a ConvNet contain more generic features (e.g. edge detectors or color blob detectors) that should be useful to many tasks, but later layers of the ConvNet becomes progressively more specific to the details of the classes contained in the original dataset. In case of ImageNet for example, which contains many dog breeds, a significant portion of the representational power of the ConvNet may be devoted to features that are specific to differentiating between dog breeds.

In our case of dataset, we can use any of pretrained architectures and fine-tune the model by unfreezing last or last 2 convolution blocks depending on size of dataset and compute power avaliable to training. This technique provides a further boost in performance than previous feature extractor technique in orders of magnitutde ranging from 0.5% to 3% but also take significant amount of time to train as we are unfreezing and training some of the convolution blocks of pretrained model.

### Next question would be when and how to fine-tune?

This is a function of several factors, but the two most important ones are the size of the new dataset (small or big), and its similarity to the original dataset (e.g. ImageNet-like in terms of the content of images and the classes, or very different, such as microscope images). Keeping in mind that ConvNet features are more generic in early layers and more original-dataset-specific in later layers, here are some common rules of thumb for navigating the 4 major scenarios:

1. New dataset is small and similar to original dataset

Since the data is small, it is not a good idea to fine-tune the ConvNet due to overfitting concerns. Since the data is similar to the original data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.

2. New dataset is large and similar to the original dataset. 

Since we have more data, we can have more confidence that we won’t overfit if we were to try to fine-tune through the full network.
    
3. New dataset is small but very different from the original dataset. 

Since the data is small, it is likely best to only train a linear classifier. Since the dataset is very different, it might not be best to train the classifier form the top of the network, which contains more dataset-specific features. Instead, it might work better to train the SVM classifier from activations somewhere earlier in the network.

4. New dataset is large and very different from the original dataset. 

Since the dataset is very large, we may expect that we can afford to train a ConvNet from scratch. However, in practice it is very often still beneficial to initialize with weights from a pretrained model. In this case, we would have enough data and confidence to fine-tune through the entire network.


So, my Young Padwan, you have now the full <span class='purple'>Power of Transfer Learning </span> and we will implement it below. <span class='orange'> And always remember the wise words spoken by Master Andrej Karpathy, "Don't be a hero. Instead of rolling your own architecture for a problem, you should look at whatever architecture currently works best on ImageNet, download a pretrained model and finetune it on your data. You should rarely ever have to train a ConvNet from scratch or design one from scratch."</span> Indeed transfer learning is your go to learning technique whenever images are inputs as long as we have some CoNvnet model pretrained on significantly larger dataset which somewhat resemebles or supersets our desired dataset. The performance boost and training time saved by using transfer learning is amazing and must be one of the first methods resorted.

In next post, we will focus on <span class='purple'>Power to Visualize CNN</span>.

By now you must have a concrete ideas about when to use Sequential and Functional API of any framework. So, we will stick to one such API in our implementation and adding implementation of both frameworks will take a lot of scrolling. Let's avoid that and use one framework.

## Recap

- Firstly, we recapped what a CNN is and how a CNN is trained.
- Next, we introduce what is transfer learning which led us to thinking about what a CNN learns.
- We looked at what the different layers in CNN learns and how we can leverage the same learning by transfer that knowledge to similar tasks.
- We looked at different ways to perform transfer learning and how transferring the knowledge helps faster learning. Primarily the two types of transferring learning include, first convnet as feature extractor where a pretrained model (models trained on Imagenet or any bigger dataset) is taken as base model and the last layer in case of Imagenet contains FC layer of 1000 units(neurons) for classifying input image in 1000 categories is cut off(removed) and replaced with our desired number of units(neurons or number of classes) and while training all the layers are frozen except the last layer inserted, and whole model is trained for our new dataset with desired number of classes. And second include, finetuning where we take above architecture and unfreeze some of layers in above frozen layers in pretrained model and retrain the model with same dataset to get new improved performance(better than previous feature extractor).
- We looked into different ways to fine-tune a pretrained model and when to fine-tune depending on type of dataset.
- We concluded that transfer learning saves a lot of time and energy(electricity unless you got Googlion planet to cover your bills) and is one of the best techniques out there to provide state of the art image classification.

---

# Keras


```
# load all the required libraries

import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split  # split dataset
import keras                                          # import keras with tensorflow as backend
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential            # sequential and functional api keras 
from keras.layers import Dense, Input, Conv2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, MaxPooling2D, InputLayer # dense and input layer for constructing mlp
from keras.optimizers import SGD
np.random.seed(42)
```

    Using TensorFlow backend.



```
# # use small subset of train, val and test

train_cats = os.listdir(train_cats_dir)
train_cats = random.sample(train_cats, 2000)
train_dogs = os.listdir(train_dogs_dir)
train_dogs = random.sample(train_dogs, 2000)
val_cats = os.listdir(val_cats_dir)
val_cats = random.sample(val_cats, 400)
val_dogs = os.listdir(val_dogs_dir)
val_dogs = random.sample(val_dogs, 400)
test_img = os.listdir(test_path)
test_img = random.sample(test_img, 50)

print ('New Training set images', len(train_cats)+len(train_dogs))
print ('New Validation set images', len(val_cats)+len(val_dogs))
print ('New Testing set images', len(test_img))
```

    New Training set images 4000
    New Validation set images 800
    New Testing set images 50



```
IMG_DIM = (224, 224)
```


```
train_X = [train_cats_dir+cats for cats in train_cats]
train_X = train_X + [train_dogs_dir+dogs for dogs in train_dogs]
train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_X]
train_imgs = np.array(train_imgs)
train_labels = [l.split('/')[-1].split('.')[0].strip('0123456789') for l in train_X]
train_labels = np.array(train_labels)
print ('Training shape:', train_imgs.shape, train_labels.shape) 

val_X = [val_cats_dir+cats for cats in val_cats]
val_X = val_X + [val_dogs_dir+dogs for dogs in val_dogs]
val_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in val_X]
val_imgs = np.array(val_imgs)
val_labels = [l.split('/')[-1].split('.')[0].strip('0123456789') for l in val_X]
val_labels = np.array(val_labels)
print ('Validation shape:', val_imgs.shape, val_labels.shape) 

test_X = [test_path+imgs for imgs in test_img]
test_X = random.sample(test_X, 50)
test_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in test_X]
test_imgs = np.array(test_imgs)
print ('Testing shape:', test_imgs.shape) 
```

    Training shape: (4000, 224, 224, 3) (4000,)
    Validation shape: (800, 224, 224, 3) (800,)
    Testing shape: (50, 224, 224, 3)



```
# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
val_labels_enc = le.transform(val_labels)

print(train_labels[:5], train_labels_enc[:5])
```

    ['cat' 'cat' 'cat' 'cat' 'cat'] [0 0 0 0 0]


## Visualization of data

Enough talk, show me the cats and dogs!


```
def preprocess_img(img, ax, label, train_dir):
    im = Image.open(os.path.join(train_dir, img))
    size = im.size
    ax.imshow(im)
    ax.set_title(f'{label} {size}')
```


```
train_x = os.listdir(train_cats_dir)
# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 10))

for idx in np.arange(10):
    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
    preprocess_img(train_x[idx], ax, 'cat', train_cats_dir)
    # print out the correct label for each image
```


<p align="center">
<img src='/images/transfer_learning_files/transfer_learning_keras_26_0.png' />
</p>



```
train_x = os.listdir(train_dogs_dir)
# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 10))

for idx in np.arange(10):
    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
    preprocess_img(train_x[idx], ax, 'dog', train_dogs_dir)
    # print out the correct label for each image
```

<p align="center">
<img src='/images/transfer_learning_files/transfer_learning_keras_27_0.png' />
</p>


### ConvNet as feature extractor


```
# [0-9] unique labels
batch_size = 50
num_classes = 2
epochs = 50

# input image dimensions
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)
```


```
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   zoom_range=0.3, 
                                   rotation_range=50,
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2, 
                                   shear_range=0.2, 
                                   horizontal_flip=True, 
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=80)
val_generator = val_datagen.flow(val_imgs, val_labels_enc, batch_size=16)
```


```
def pretrained_models(name):
    
    if name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, 
                           input_shape=input_shape)
 
        output = base_model.layers[-1].output
        output = Flatten()(output)
        
    model = Model(inputs=base_model.input, outputs=output) 
  
    return model

vgg_model = pretrained_models('VGG16')

vgg_model.trainable = False
for layer in vgg_model.layers:
  layer.trainable = False

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer.name, layer.trainable) for layer in vgg_model.layers]
pd.DataFrame(layers, columns=['Layer Name', 'Layer Trainable'])  
```

    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58892288/58889256 [==============================] - 1s 0us/step





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Layer Name</th>
      <th>Layer Trainable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>input_1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>block1_conv1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>block1_conv2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>block1_pool</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>block2_conv1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>block2_conv2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>block2_pool</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>block3_conv1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>block3_conv2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>block3_conv3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>block3_pool</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>block4_conv1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>block4_conv2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>block4_conv3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>block4_pool</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>block5_conv1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>block5_conv2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>block5_conv3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>block5_pool</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>flatten_1</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```
input_shape = vgg_model.output_shape[1]

model = Sequential()
model.add(vgg_model)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    model_1 (Model)              (None, 25088)             14714688  
    _________________________________________________________________
    dense_1 (Dense)              (None, 512)               12845568  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 512)               262656    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 513       
    =================================================================
    Total params: 27,823,425
    Trainable params: 13,108,737
    Non-trainable params: 14,714,688
    _________________________________________________________________



```
history = model.fit_generator(train_generator, 
                              steps_per_epoch=50, 
                              epochs=epochs,
                              validation_data=val_generator, 
                              validation_steps=50) 
```

    Epoch 1/50
    50/50 [==============================] - 71s 1s/step - loss: 0.8805 - acc: 0.6373 - val_loss: 0.3111 - val_acc: 0.8612
    Epoch 2/50
    50/50 [==============================] - 60s 1s/step - loss: 0.4145 - acc: 0.8145 - val_loss: 0.2918 - val_acc: 0.8650
    Epoch 3/50
    50/50 [==============================] - 61s 1s/step - loss: 0.3371 - acc: 0.8492 - val_loss: 0.2929 - val_acc: 0.8712
    Epoch 4/50
    50/50 [==============================] - 60s 1s/step - loss: 0.3755 - acc: 0.8210 - val_loss: 0.2396 - val_acc: 0.8888
    Epoch 5/50
    50/50 [==============================] - 61s 1s/step - loss: 0.3447 - acc: 0.8423 - val_loss: 0.2149 - val_acc: 0.9175
    Epoch 6/50
    50/50 [==============================] - 60s 1s/step - loss: 0.3153 - acc: 0.8640 - val_loss: 0.1894 - val_acc: 0.9225
    Epoch 7/50
    50/50 [==============================] - 60s 1s/step - loss: 0.3164 - acc: 0.8600 - val_loss: 0.2024 - val_acc: 0.9187
    Epoch 8/50
    50/50 [==============================] - 60s 1s/step - loss: 0.3190 - acc: 0.8600 - val_loss: 0.2002 - val_acc: 0.9175
    Epoch 9/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2998 - acc: 0.8640 - val_loss: 0.2040 - val_acc: 0.9163
    Epoch 10/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2944 - acc: 0.8715 - val_loss: 0.1984 - val_acc: 0.9125
    Epoch 11/50
    50/50 [==============================] - 61s 1s/step - loss: 0.3289 - acc: 0.8510 - val_loss: 0.2137 - val_acc: 0.9250
    Epoch 12/50
    50/50 [==============================] - 61s 1s/step - loss: 0.3104 - acc: 0.8617 - val_loss: 0.1910 - val_acc: 0.9213
    Epoch 13/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2989 - acc: 0.8723 - val_loss: 0.1815 - val_acc: 0.9275
    Epoch 14/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2942 - acc: 0.8657 - val_loss: 0.1970 - val_acc: 0.9187
    Epoch 15/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2808 - acc: 0.8825 - val_loss: 0.2038 - val_acc: 0.9125
    Epoch 16/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2948 - acc: 0.8705 - val_loss: 0.1876 - val_acc: 0.9250
    Epoch 17/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2816 - acc: 0.8772 - val_loss: 0.1874 - val_acc: 0.9275
    Epoch 18/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2794 - acc: 0.8768 - val_loss: 0.1775 - val_acc: 0.9300
    Epoch 19/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2708 - acc: 0.8812 - val_loss: 0.1897 - val_acc: 0.9213
    Epoch 20/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2696 - acc: 0.8840 - val_loss: 0.1866 - val_acc: 0.9300
    Epoch 21/50
    50/50 [==============================] - 61s 1s/step - loss: 0.2720 - acc: 0.8775 - val_loss: 0.1997 - val_acc: 0.9263
    Epoch 22/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2708 - acc: 0.8807 - val_loss: 0.1777 - val_acc: 0.9337
    Epoch 23/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2663 - acc: 0.8845 - val_loss: 0.1934 - val_acc: 0.9175
    Epoch 24/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2701 - acc: 0.8777 - val_loss: 0.1710 - val_acc: 0.9337
    Epoch 25/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2845 - acc: 0.8753 - val_loss: 0.1993 - val_acc: 0.9137
    Epoch 26/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2826 - acc: 0.8765 - val_loss: 0.1687 - val_acc: 0.9250
    Epoch 27/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2785 - acc: 0.8800 - val_loss: 0.1716 - val_acc: 0.9313
    Epoch 28/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2941 - acc: 0.8692 - val_loss: 0.1710 - val_acc: 0.9250
    Epoch 29/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2738 - acc: 0.8767 - val_loss: 0.1851 - val_acc: 0.9175
    Epoch 30/50
    50/50 [==============================] - 61s 1s/step - loss: 0.2774 - acc: 0.8815 - val_loss: 0.2411 - val_acc: 0.9012
    Epoch 31/50
    50/50 [==============================] - 61s 1s/step - loss: 0.2896 - acc: 0.8692 - val_loss: 0.1796 - val_acc: 0.9325
    Epoch 32/50
    50/50 [==============================] - 61s 1s/step - loss: 0.2606 - acc: 0.8827 - val_loss: 0.1772 - val_acc: 0.9313
    Epoch 33/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2440 - acc: 0.8962 - val_loss: 0.1727 - val_acc: 0.9287
    Epoch 34/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2972 - acc: 0.8645 - val_loss: 0.1890 - val_acc: 0.9237
    Epoch 35/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2464 - acc: 0.8975 - val_loss: 0.1734 - val_acc: 0.9263
    Epoch 36/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2789 - acc: 0.8735 - val_loss: 0.1742 - val_acc: 0.9325
    Epoch 37/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2821 - acc: 0.8755 - val_loss: 0.1747 - val_acc: 0.9287
    Epoch 38/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2726 - acc: 0.8780 - val_loss: 0.1731 - val_acc: 0.9263
    Epoch 39/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2693 - acc: 0.8800 - val_loss: 0.1896 - val_acc: 0.9200
    Epoch 40/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2715 - acc: 0.8715 - val_loss: 0.1773 - val_acc: 0.9263
    Epoch 41/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2521 - acc: 0.8877 - val_loss: 0.1700 - val_acc: 0.9313
    Epoch 42/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2442 - acc: 0.8945 - val_loss: 0.1736 - val_acc: 0.9375
    Epoch 43/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2552 - acc: 0.8865 - val_loss: 0.1925 - val_acc: 0.9250
    Epoch 44/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2516 - acc: 0.8863 - val_loss: 0.1789 - val_acc: 0.9413
    Epoch 45/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2498 - acc: 0.8910 - val_loss: 0.1749 - val_acc: 0.9275
    Epoch 46/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2520 - acc: 0.8873 - val_loss: 0.1786 - val_acc: 0.9300
    Epoch 47/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2518 - acc: 0.8947 - val_loss: 0.1664 - val_acc: 0.9363
    Epoch 48/50
    50/50 [==============================] - 61s 1s/step - loss: 0.2589 - acc: 0.8815 - val_loss: 0.1873 - val_acc: 0.9275
    Epoch 49/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2603 - acc: 0.8865 - val_loss: 0.1737 - val_acc: 0.9387
    Epoch 50/50
    50/50 [==============================] - 60s 1s/step - loss: 0.2453 - acc: 0.8943 - val_loss: 0.1733 - val_acc: 0.9350



```
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

<p align="center">
<img src='/images/transfer_learning_files/transfer_learning_keras_34_0.png' />
</p>


<p align="center">
<img src='/images/transfer_learning_files/transfer_learning_keras_34_1.png' />
</p>



```
test_predictions = model.predict_on_batch(test_imgs/225.)
print (test_predictions.shape)
```

    (50, 1)



```
# obtain one batch of test images
images, predict = test_imgs, test_predictions

# convert output probabilities to predicted class
preds = (predict > 0.5).astype('int')

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(array_to_img(images[idx]))
    if preds[idx] == 0:
        test_predictions[idx] = 1-test_predictions[idx]
    ax.set_title("{:.2f} % Accuracy {}".format(float(test_predictions[idx][0]*100), 'cat' if preds[idx]==0 else 'dog'))
```


<p align="center">
<img src='/images/transfer_learning_files/transfer_learning_keras_36_0.png' />
</p>



```
v_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow(val_imgs, val_labels_enc, batch_size=batch_size)
img, lbl = val_generator.next()

v_predictions = model.predict_on_batch(img)
print (v_predictions.shape, lbl.shape)
print (v_predictions[:5], lbl[:5])
```

    (50, 1) (50,)
    [[0.5877145 ]
     [1.        ]
     [0.9941321 ]
     [0.98659295]
     [0.00550673]] [1 1 1 1 1]



```
# obtain one batch of test images
images, predict = img, lbl

# convert output probabilities to predicted class
pred = (predict > 0.5).astype('int')
preds = le.inverse_transform(pred)
labels = le.inverse_transform(lbl)

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(array_to_img(images[idx]))
    ax.set_title("{} ({})".format(str(preds[idx]), str(labels[idx])),
                 color=("green" if preds[idx]==labels[idx] else "red"))
```


<p align="center">
<img src='/images/transfer_learning_files/transfer_learning_keras_38_0.png' />
</p>



```
val_preds = model.predict(val_imgs, batch_size=batch_size)
print (val_preds.shape, val_labels_enc.shape)
```

    (800, 1) (800,)



```
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(val_labels_enc, val_preds.astype('int'), normalize=False)
```

<p align="center">
<img src='/images/transfer_learning_files/transfer_learning_keras_40_1.png' />
</p>



```
model.save('bottleneck-features.h5')
```

### Fine tuning


```
for i, layer in enumerate(vgg_model.layers):
    print (i, layer.name, layer.trainable)
    
for i, layer in enumerate(model.layers):
    print (i, layer.name, layer.trainable)
```

    0 input_1 False
    1 block1_conv1 False
    2 block1_conv2 False
    3 block1_pool False
    4 block2_conv1 False
    5 block2_conv2 False
    6 block2_pool False
    7 block3_conv1 False
    8 block3_conv2 False
    9 block3_conv3 False
    10 block3_pool False
    11 block4_conv1 False
    12 block4_conv2 False
    13 block4_conv3 False
    14 block4_pool False
    15 block5_conv1 False
    16 block5_conv2 False
    17 block5_conv3 False
    18 block5_pool False
    19 flatten_1 False
    0 model_1 False
    1 dense_1 True
    2 dropout_1 True
    3 dense_2 True
    4 dropout_2 True
    5 dense_3 True



```
# we chose to train the top 1 convolution block, i.e. we will freeze
# the first 15 layers and unfreeze the rest:
for layer in vgg_model.layers[:11]:
    layer.trainable = False
for layer in vgg_model.layers[11:]:
    layer.trainable = True

for i, layer in enumerate(vgg_model.layers):
    print (i, layer.name, layer.trainable)
    
    
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.0001, momentum=0.9),
              metrics=['accuracy'])
                 
print (model.summary())
```

    0 input_1 False
    1 block1_conv1 False
    2 block1_conv2 False
    3 block1_pool False
    4 block2_conv1 False
    5 block2_conv2 False
    6 block2_pool False
    7 block3_conv1 False
    8 block3_conv2 False
    9 block3_conv3 False
    10 block3_pool False
    11 block4_conv1 True
    12 block4_conv2 True
    13 block4_conv3 True
    14 block4_pool True
    15 block5_conv1 True
    16 block5_conv2 True
    17 block5_conv3 True
    18 block5_pool True
    19 flatten_1 True
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    model_1 (Model)              (None, 25088)             14714688  
    _________________________________________________________________
    dense_1 (Dense)              (None, 512)               12845568  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 512)               262656    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 513       
    =================================================================
    Total params: 27,823,425
    Trainable params: 13,108,737
    Non-trainable params: 14,714,688
    _________________________________________________________________
    None



```
history = model.fit_generator(train_generator, 
                              steps_per_epoch=50, 
                              epochs=20,
                              validation_data=val_generator, 
                              validation_steps=50) 
```

    Epoch 1/20
    50/50 [==============================] - 83s 2s/step - loss: 0.2441 - acc: 0.8890 - val_loss: 0.1756 - val_acc: 0.9336
    Epoch 2/20
    50/50 [==============================] - 74s 1s/step - loss: 0.2423 - acc: 0.8928 - val_loss: 0.1749 - val_acc: 0.9328
    Epoch 3/20
    50/50 [==============================] - 74s 1s/step - loss: 0.2305 - acc: 0.8990 - val_loss: 0.1692 - val_acc: 0.9340
    Epoch 4/20
    50/50 [==============================] - 74s 1s/step - loss: 0.2361 - acc: 0.8977 - val_loss: 0.1740 - val_acc: 0.9276
    Epoch 5/20
    50/50 [==============================] - 74s 1s/step - loss: 0.2211 - acc: 0.9025 - val_loss: 0.1799 - val_acc: 0.9224
    Epoch 6/20
    50/50 [==============================] - 74s 1s/step - loss: 0.2389 - acc: 0.8950 - val_loss: 0.1687 - val_acc: 0.9348
    Epoch 7/20
    50/50 [==============================] - 74s 1s/step - loss: 0.2377 - acc: 0.8930 - val_loss: 0.1799 - val_acc: 0.9304
    Epoch 8/20
    50/50 [==============================] - 73s 1s/step - loss: 0.2223 - acc: 0.9048 - val_loss: 0.1756 - val_acc: 0.9296
    Epoch 9/20
    50/50 [==============================] - 73s 1s/step - loss: 0.2224 - acc: 0.9038 - val_loss: 0.1786 - val_acc: 0.9292
    Epoch 10/20
    50/50 [==============================] - 74s 1s/step - loss: 0.2282 - acc: 0.9042 - val_loss: 0.1745 - val_acc: 0.9300
    Epoch 11/20
    50/50 [==============================] - 73s 1s/step - loss: 0.2279 - acc: 0.9007 - val_loss: 0.1709 - val_acc: 0.9328
    Epoch 12/20
    50/50 [==============================] - 74s 1s/step - loss: 0.2212 - acc: 0.9045 - val_loss: 0.1815 - val_acc: 0.9232
    Epoch 13/20
    50/50 [==============================] - 73s 1s/step - loss: 0.2376 - acc: 0.8950 - val_loss: 0.1644 - val_acc: 0.9328
    Epoch 14/20
    50/50 [==============================] - 74s 1s/step - loss: 0.2226 - acc: 0.9012 - val_loss: 0.1711 - val_acc: 0.9304
    Epoch 15/20
    50/50 [==============================] - 73s 1s/step - loss: 0.2271 - acc: 0.9010 - val_loss: 0.1772 - val_acc: 0.9288
    Epoch 16/20
    50/50 [==============================] - 73s 1s/step - loss: 0.2334 - acc: 0.8975 - val_loss: 0.1728 - val_acc: 0.9284
    Epoch 17/20
    50/50 [==============================] - 74s 1s/step - loss: 0.2217 - acc: 0.9078 - val_loss: 0.1770 - val_acc: 0.9244
    Epoch 18/20
    50/50 [==============================] - 73s 1s/step - loss: 0.2311 - acc: 0.8915 - val_loss: 0.1694 - val_acc: 0.9284
    Epoch 19/20
    50/50 [==============================] - 73s 1s/step - loss: 0.2257 - acc: 0.9002 - val_loss: 0.1755 - val_acc: 0.9252
    Epoch 20/20
    50/50 [==============================] - 73s 1s/step - loss: 0.2298 - acc: 0.8950 - val_loss: 0.1602 - val_acc: 0.9340



```
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

<p align="center">
<img src='/images/transfer_learning_files/transfer_learning_keras_46_0.png' />
</p>


<p align="center">
<img src='/images/transfer_learning_files/transfer_learning_keras_46_1.png' />
</p>



```
test_predictions = model.predict_on_batch(test_imgs)
print (test_predictions.shape)
```

    (50, 1)



```
# obtain one batch of test images
images, predict = test_imgs, test_predictions

# convert output probabilities to predicted class
preds = (predict > 0.5).astype('int')

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(array_to_img(images[idx]))
    if preds[idx] == 0:
        test_predictions[idx] = 1-test_predictions[idx]
    ax.set_title("{:.2f} % Accuracy {}".format(float(test_predictions[idx][0]*100), 'cat' if preds[idx][0]==0 else 'dog'))
```


<p align="center">
<img src='/images/transfer_learning_files/transfer_learning_keras_48_0.png' />
</p>



```
model.save('finetune.h5')
```


```
v_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow(val_imgs, val_labels_enc, batch_size=batch_size)
img, lbl = val_generator.next()

v_predictions = model.predict_on_batch(img)
print (v_predictions.shape, lbl.shape)
print (v_predictions[:5], lbl[:5])
```

    (50, 1) (50,)
    [[8.9467996e-01]
     [3.6521584e-01]
     [4.8128858e-01]
     [1.9999747e-01]
     [5.2365294e-04]] [0 1 1 1 0]



```
# obtain one batch of test images
images, predict = img, lbl

# convert output probabilities to predicted class
pred = (predict > 0.5).astype('int')
preds = le.inverse_transform(pred)
labels = le.inverse_transform(lbl)

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(array_to_img(images[idx]))
    ax.set_title("{} ({})".format(str(preds[idx]), str(labels[idx])),
                 color=("green" if preds[idx]==labels[idx] else "red"))
```

<p align="center">
<img src='/images/transfer_learning_files/transfer_learning_keras_51_0.png' />
</p>


```
val_preds = model.predict(val_imgs, batch_size=batch_size)
print (val_preds.shape, val_labels_enc.shape)
```

    (800, 1) (800,)



```
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(val_labels_enc, val_preds.astype('int'), normalize=False)
```

<p align="center">
<img src='/images/transfer_learning_files/transfer_learning_keras_53_1.png' />
</p>


<font color='red'>I-know-everything:</font> Young Padwan, now that you have seen how Transfer Learning works, in next post we will visualize layers in CNN and see what parts of image are they looking at. Visualization layers in CNN plays a crucial role in seeing what is going inside the black box of CNN. Some of the popular visualization techniques include:

- Gradient visualization
- Smooth grad
- CNN filter visualization
- Inverted image representations
- Deep dream
- Class specific image generation

<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology


Power of Transfer Learning - Transfer Learning

Power of Visualize CNN - Visualize CNN

ConvNets - Convolution Neural Networks

neurons - unit

---

# Further Reading

[Sebastian Ruder's blog on Transfer Learning](http://ruder.io/transfer-learning/)

[Visualizaing and Understanding Convolution Neural Networks](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)

---

# Footnotes and Credits

[Kaggle Dataset for Cats and Dogs](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

[Traditional ML setup](http://ruder.io/transfer-learning/)

[Cat and Dog image](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

[Transfer Learning graphic](https://medium.com/the-official-integrate-ai-blog/transfer-learning-explained-7d275c1e34e2)

[3 ways in which learning is imporved by transfer](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)

[Visualize CNN](http://cs231n.stanford.edu/slides/winter1516_lecture7.pdf)

[Different examples of layers in CNN](https://stats.stackexchange.com/questions/146413/why-convolutional-neural-networks-belong-to-deep-learning)

[Visualize layers in CNN](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)
