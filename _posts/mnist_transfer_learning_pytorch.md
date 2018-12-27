

```python
# import shutil 
# shutil.rmtree('data/catsvsdogs')
```

# Transfer Learning

In this notebook, we will go through basics of Transfer Learning and what a CNN visualizes when we pass image to model. We will implement this using two popular deep learning frameworks `Keras` and `PyTorch`. 

Hey yo, but what is Transfer Learning?

Well sit tight and buckle up. I will go through everything in-detail.

# Getting Data


The cats vs dogs dataset isn't available on keras library. You can download it from Kaggle however. Let's see how to do this by using the Kaggle API as it's going to be pretty useful to you if you want to join a competition or use other Kaggle datasets later on.

First, install the Kaggle API by uncommenting the following line and executing it, or by executing it in your terminal.


```python
!pip install --upgrade kaggle
```



Then you need to upload your credentials from Kaggle on your instance. Login to kaggle and click on your profile picture on the top left corner, then 'My account'. Scroll down until you find a button named 'Create New API Token' and click on it. This will trigger the download of a file named 'kaggle.json'.

Upload this file to the directory this notebook is running in, by clicking "Upload" on your main Jupyter page, then uncomment and execute the next two commands (or run them in a terminal).



```python
! mkdir -p ~/.kaggle/
! mv kaggle.json ~/.kaggle/
```

You're all set to download the data from [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) competition. You first need to go to its main page and accept its rules, and run the two cells below (uncomment the shell commands to download and unzip the data). If you get a 403 forbidden error it means you haven't accepted the competition rules yet (you have to go to the competition page, click on Rules tab, and then scroll to the bottom to find the accept button).



```python
! kaggle competitions download -c dogs-vs-cats-redux-kernels-edition -p 'data/catsvsdogs'
```


```python
path = 'data/catsvsdogs/'

! unzip -q -n {path}/train.zip -d {path}
! unzip -q -n {path}/test.zip -d {path}
```


```python
train_path = 'data/catsvsdogs/train/'
val_path = 'data/catsvsdogs/val/'
test_path = 'data/catsvsdogs/test/'
train_cats_dir = f'{train_path}cats/'
train_dogs_dir = f'{train_path}dogs/'
val_cats_dir = f'{val_path}cats/'
val_dogs_dir = f'{val_path}dogs/'
```


```python
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
%matplotlib inline
```


```python
print ('Training set images', len(os.listdir(train_path)))
print ('Test set images', len(os.listdir(test_path)))
```

    Training set images 2
    Test set images 12500



data/
    train/
        dog001.jpg
        dog002.jpg
        ...
        cat001.jpg
        cat002.jpg
        ...
    test/
        001.jpg
        002.jpg
        ...



```python
train_imgs = os.listdir(train_path)
train_cats_dir = f'{train_path}cats/'
train_dogs_dir = f'{train_path}dogs/'
os.makedirs(train_cats_dir)
os.makedirs(train_dogs_dir)
print ('[INFO] Train Folder for dogs and cats created....')

print ('[INFO] Moving train images to dogs and cats folders....')
for img in tqdm(train_imgs):
    ex = img.split('.')
    new_img = ex[0]+ex[1]+'.'+ex[2]
    if ex[0] == 'dog':
        os.rename(f'{train_path}{img}', f'{train_dogs_dir}{new_img}')
    else:
        os.rename(f'{train_path}{img}', f'{train_cats_dir}{new_img}')   
print ('[INFO] Moving images from train to cats and dogs complete... ')   
```



data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    test/
        001.jpg
        002.jpg
        003.jpg
        004.jpg
        ...


```python
# create validation set from 20% of training set sampled randomly

val_path = 'data/catsvsdogs/val/'
train_cat_imgs = os.listdir(train_cats_dir)
train_dog_imgs = os.listdir(train_dogs_dir)
os.makedirs(val_path)
print ('[INFO] Val Folder created....')
val_cats_dir = f'{val_path}cats/'
val_dogs_dir = f'{val_path}dogs/'
os.makedirs(val_cats_dir)
os.makedirs(val_dogs_dir)
print ('[INFO] Val Folder for dogs and cats created....')

print ('[INFO] Random sample 20% of cats from train to val...')
val_size = 0.2
trn_cat_imgs = os.listdir(train_cats_dir)
val_cat_len = int(len(trn_cat_imgs) * 0.2)
val_cat_imgs = random.sample(trn_cat_imgs, val_cat_len)

for img in tqdm(val_cat_imgs):
    os.rename(f'{train_cats_dir}{img}', f'{val_cats_dir}{img}')  
print ('[INFO] Moving images from train cat to val cat complete...')   

print ('[INFO] Random sample 20% of dogs from train to val...')
val_size = 0.2
trn_dog_imgs = os.listdir(train_dogs_dir)
val_dog_len = int(len(trn_dog_imgs) * 0.2)
val_dog_imgs = random.sample(trn_dog_imgs, val_dog_len)

for img in tqdm(val_dog_imgs):
    os.rename(f'{train_dogs_dir}{img}', f'{val_dogs_dir}{img}')  
print ('[INFO] Moving images from train dog to val dog complete... ')   

```


```python
print ('Training set images', len(os.listdir(train_cats_dir))+len(os.listdir(train_dogs_dir)))
print ('Validation set images', len(os.listdir(val_cats_dir))+len(os.listdir(val_dogs_dir)))
print ('Test set images', len(os.listdir(test_path)))
```

    Training set images 20000
    Validation set images 5000
    Test set images 12500


## Visualization of data

Enough talk, show me the data!


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


![png](mnist_transfer_learning_pytorch_files/mnist_transfer_learning_pytorch_19_0.png)



```python
train_x = os.listdir(train_dogs_dir)
# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 10))

for idx in np.arange(10):
    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
    preprocess_img(train_x[idx], ax, 'dog', train_dogs_dir)
    # print out the correct label for each image
```


![png](mnist_transfer_learning_pytorch_files/mnist_transfer_learning_pytorch_20_0.png)


# Learning curves

Lets dive into interpreting learning different curves to understand and ways to avoid underfitting-overfitting or bias-variance tradeoff. There is some sort of tug of war between bias and variance, if we reduce bias error that leads to increase in variance error.

Let's recap what we had from our previous discussion on bias and variance.

- (Training - Dev) error high ==> High variance ==> Overfitting ==> Add more data to training set
- Training error high     ==> High bias    ===> Underfitting ==> Make model more complex
- Bayes error ==> Optimal Rate  ==> Unavoidable bias
- (Training - Bayes) error ===> Avoidable bias
- Bias = Optimal error rate (“unavoidable bias”) + Avoidable bias


## Cat Classifier

Back to cats, suppose we run the algorithm using different training set sizes. For example, if you have 1,000 examples, we train separate copies of the algorithm on 100, 200, 300, ..., 1000 examples. Following are the different learning curves, where desired performance(green) along with dev(red) error and train(blue) error are plotted against the number of training examples.


Consider this learning curve,

high_variance_bias.png

Is this plot indicating, high bias, high variance or both?

The training error is very close to desired performance, indicating avoidable bias is very low. The training(blue) error curve is relatively low, and dev(red) error is much higher than training error. Thus, the bias is small, but variance is large. As from recap above, adding more training data will help close gap between training and dev error and help reduce high variance.



Consider this curve,

significant_variance_bias.png

Is this plot indicating, high bias, high variance or both?

This time, training error is large, as it is much higher than desired performance. There is significant avoidable bias. The dev error is also much larger than training error. This indicated we have significant bias and significant variance in our plot. We will use the ways to avoid both variance and bias.



Consider this curve,

significant_variance_bias.png

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




In next post, we will go into plotting learning curves and how to interpret them. Stay tuned!

# Introduction to Transfer Learning





<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<span class='red'>I-know-everything:</span> Yo, apperentice. This time you will experience the <span class="purple"> Power of Transfer Learning</span>. Transfer Learning is a technique where you take a pretrained model trained on large dataset and transfer the learned knowledge to another model with small dataset but some what similar to large dataset for classification. For e.g. if we consider Imagenet dataset which contains 1.2 million images and 1000 categories, in that there are 24 different categories of dogs and 16 different categories of cats. So, we can transfer the learned features of cats and dogs from model trained on Imagenet dataset to our new model which contains 25,000 images of dogs and cats in training set.

traditional_ml_setup.png
http://ruder.io/transfer-learning/

transfer_learning_steup.png

transfer_learning_1.png
credits: https://medium.com/the-official-integrate-ai-blog/transfer-learning-explained-7d275c1e34e2


<span class='green'>I-know-nothing:</span> Ahh?

<span class='red'>I-know-everything:</span> Okay, let's take a step back and go over our learning from <span class='purple'>Force of CNN</span>. First, we saw what a convolution operator is, how different kernels or the numbers i n matrix give differnet results when applied to an image such as edge detector, blurring, sharpening, etc. After that, we visited different functions and looked at their properties and role in CNN, e.g. kernel, pooling, strides. We saw CNN consists of multiple CONV-RELU-POOL layers, followed by FC layers like the one shown below.

tesla_cs231n.png

We saw how the training a CNN is similar to MLP. It consists of forward pass followed by backward pass where the kernels adjust the weights so as to backpropogate the error in classification and also looked at different architectures and role they played in Imagenet competition. The only thing we did not discuss is that what these CNN are learning that makes them able to classify 1.2 million images in 1000 categories with 2.25% top5 error rate better than humans. What is going on insides these layers to them such better classifiers?

visualize_cnn.png
http://cs231n.stanford.edu/slides/winter1516_lecture7.pdf

Many details of how these models works is still a mystery (black-box), but Zeiler and Fergus showed in their excellent [paper](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf) on Visualizaing and Understanding Convolution Neural Networks, that lower convolutional layers capture low-level image features, e.g. edges, while higher convolutional layers capture more and more complex details, such as body parts, faces, and other compositional features.

The final fully-connected layers are generally assumed to capture information that is relevant for solving the respective task, e.g. AlexNet's fully-connected layers would indicate which features are relevant to classify an image into one of 1000 object categories.

layer1_layer2

layer3

layer4_layer5

credits: https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf

As we observe in above pictures, different layers correspond or activate to different features in the images. For e.g., Layer 3 activates for different textures, Layer 2 activates for different edges and circles, similarly, Layer 5 activates for faces of humans, animals also, they learn to identify text in the image on their own.

In short, here is how CNN learns.

layers_cnn.jpg
https://stats.stackexchange.com/questions/146413/why-convolutional-neural-networks-belong-to-deep-learning

When an image of face of human is passed through CNN, the initial layers learn to identify simple features like nose, eyes, ears, etc. As we move up the architecture, the higher layers will combine simple features into more complex feature and finally dense layers at the top of the network will combine very high level features and produce classification predictions.

<span class='green'>I-know-nothing:</span> Now I understand what goes behind the scenes of CNN model. So, how can these features help us in training our model?

<span class='red'>I-know-everything:</span> Glad you asked. Transfer learning is an optimization, a shortcut to saving time or getting better performance. There are tthree possible benefits to look for when using transfer learning:

1. Higher start. The initial skill (before refining the model) on the source model is higher than it otherwise would be.
2. Higher slope. The rate of improvement of skill during training of the source model is steeper than it otherwise would be.
3. Higher asymptote. The converged skill of the trained model is better than it otherwise would be.

Three-ways-in-which-transfer-might-improve-learning.png
credits: https://machinelearningmastery.com/transfer-learning-for-deep-learning/

On some problems where you may not have very much data, transfer learning can enable you to develop skillful models that you simply could not develop in the absence of transfer learning.

For a new classification task, we can simply use the off-the-shelf features of a state-of-the-art CNN pre-trained on ImageNet and train a new model on these extracted features. 

There are 2 major Transfer Learning scenarios:

1. ConvNet as fixed feature extractor

Take a ConvNet pretrained on ImageNet, remove the last fully-connected layer (this layer’s outputs are the 1000 class scores for a different task like ImageNet), then treat the rest of the ConvNet as a fixed feature extractor for the new dataset. For eg, in AlexNet, this would compute a 4096-D vector for every image that contains the activations of the hidden layer immediately before the classifier. We call these features <span class='orange'>CNN codes</span>. Once you extract the 4096-D codes for all images, train a linear classifier (e.g. Linear SVM or Softmax classifier) for the new dataset.

2. Finetuning the ConvNet

The second strategy is to not only replace and retrain the classifier on top of the ConvNet on the new dataset, but to also fine-tune the weights of the pretrained network by continuing the backpropagation. It is possible to fine-tune all the layers of the ConvNet, or it’s possible to keep some of the earlier layers fixed (due to overfitting concerns) and only fine-tune some higher-level portion of the network. This is motivated by the observation that the earlier features of a ConvNet contain more generic features (e.g. edge detectors or color blob detectors) that should be useful to many tasks, but later layers of the ConvNet becomes progressively more specific to the details of the classes contained in the original dataset. In case of ImageNet for example, which contains many dog breeds, a significant portion of the representational power of the ConvNet may be devoted to features that are specific to differentiating between dog breeds.


##### Next question would be when and how to fine-tune?

This is a function of several factors, but the two most important ones are the size of the new dataset (small or big), and its similarity to the original dataset (e.g. ImageNet-like in terms of the content of images and the classes, or very different, such as microscope images). Keeping in mind that ConvNet features are more generic in early layers and more original-dataset-specific in later layers, here are some common rules of thumb for navigating the 4 major scenarios:

1. New dataset is small and similar to original dataset

Since the data is small, it is not a good idea to fine-tune the ConvNet due to overfitting concerns. Since the data is similar to the original data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.

2. New dataset is large and similar to the original dataset. 

Since we have more data, we can have more confidence that we won’t overfit if we were to try to fine-tune through the full network.
    
3. New dataset is small but very different from the original dataset. 

Since the data is small, it is likely best to only train a linear classifier. Since the dataset is very different, it might not be best to train the classifier form the top of the network, which contains more dataset-specific features. Instead, it might work better to train the SVM classifier from activations somewhere earlier in the network.

4. New dataset is large and very different from the original dataset. 

Since the dataset is very large, we may expect that we can afford to train a ConvNet from scratch. However, in practice it is very often still beneficial to initialize with weights from a pretrained model. In this case, we would have enough data and confidence to fine-tune through the entire network.


So, my Young Padwan, you have now the full <span class='purple'>Power of Transfer Learning </span> and we will implement it below. <span class='orange'> And always remember the wise words spoken by Master Andrej Karpathy, "Don't be a hero. Instead of rolling your own architecture for a problem, you should look at whatever architecture currently works best on ImageNet, download a pretrained model and finetune it on your data. You should rarely ever have to train a ConvNet from scratch or design one from scratch."</span>

Next, we will focus on <span class='purple'> Power to Visualize CNN </span>.

By now you must have a concrete ideas about when to use Sequential and Functional API of any framework. So, we will stick to one such API in our implementation.


## PyTorch


```python
# load all the required libraries

import random
import torch                                          # import keras with tensorflow as backend
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as data
from torch import optim

np.random.seed(42)
```


```python
class TestImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        images = []
        for filename in os.listdir(root):
            if filename.endswith('jpg'):
                images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)
    
class LimitDataset(data.Dataset):
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.dataset[i]
```


```python
batch_size = 20
num_workers = 0

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/catsvsdogs/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

image_datasets['test'] = TestImageFolder(os.path.join(data_dir, 'test'),
                                          data_transforms['test'])

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
print (dataset_sizes)

# use small subset of train, val and test
num_train = 200
num_val = 80
num_test = 20
train_indices = np.random.choice(np.arange(dataset_sizes['train']), size=num_train)
val_indices = np.random.choice(np.arange(dataset_sizes['val']), size=num_val)
test_indices = np.random.choice(np.arange(dataset_sizes['test']), size=num_test)

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, 
    sampler=val_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, 
    sampler=test_sampler, num_workers=num_workers)

print('Training samples:', len(train_loader.dataset))
print ('Validation samples:', len(val_loader.dataset))
print('Test samples:', len(test_loader.dataset))

# Create training and validation dataloaders
dataloaders = {'train': train_loader, 
               'val': val_loader}

class_names = image_datasets['train'].classes
print ('Classes:', class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

    {'train': 20000, 'val': 5000, 'test': 12500}
    Training samples: 20000
    Validation samples: 5000
    Test samples: 12500
    Classes: ['cats', 'dogs']



```python
test_sampler.indices
```




    array([ 4811, 11939,  8954,  5346,  5043,  2428, 10699,  5126,  8220,
            9098, 11296,  7813, 12191, 11186, 10698, 11801, 11745,  9645,
            3356,  8707])




```python
print (image_datasets)
```

    {'train': Dataset ImageFolder
        Number of datapoints: 20000
        Root Location: data/catsvsdogs/train
        Transforms (if any): Compose(
                                 RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BILINEAR)
                                 RandomHorizontalFlip(p=0.5)
                                 ToTensor()
                                 Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             )
        Target Transforms (if any): None, 'val': Dataset ImageFolder
        Number of datapoints: 5000
        Root Location: data/catsvsdogs/val
        Transforms (if any): Compose(
                                 Resize(size=256, interpolation=PIL.Image.BILINEAR)
                                 CenterCrop(size=(224, 224))
                                 ToTensor()
                                 Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             )
        Target Transforms (if any): None, 'test': <__main__.TestImageFolder object at 0x7f4815c4ef60>}



```python
# # use small subset of train, val and test

# train_cats = os.listdir(train_cats_dir)
# train_dogs = os.listdir(train_dogs_dir)
# val_cats = os.listdir(val_cats_dir)
# val_dogs = os.listdir(val_dogs_dir)
# test_imgs = os.listdir(test_path)

# print ('Training set images', len(train_cats)+len(train_dogs))
# print ('Validation set images', len(val_cats)+len(val_dogs))
# print ('Testing set images', len(test_imgs))
```

    Training set images 20000
    Validation set images 5000
    Testing set images 12500


### ConvNet as feature extractor


```python
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()

    history = dict()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
         # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            
                        running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / (len(dataloaders[phase])*batch_size)
            epoch_acc = running_corrects.double() / (len(dataloaders[phase])*batch_size)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print (len(dataloaders[phase].dataset))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase+'_acc' in history:
                # append the new number to the existing array at this slot
                history[phase+'_acc'].append(epoch_acc)
            else:
                # create a new array in this slot
                history[phase+'_acc'] = [epoch_acc]
            
            if phase+'_loss' in history:
                # append the new number to the existing array at this slot
                history[phase+'_loss'].append(epoch_loss)
            else:
                # create a new array in this slot
                history[phase+'_loss'] = [epoch_loss]            

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history
```


```python
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)
print (model_conv)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

#number of epochs
epochs = 1
```


```python
model, history = train_model(model_conv, dataloaders, criterion, optimizer, epochs)
```


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


```python

```


```python
test_predictions = model.predict(test_x)
print (test_predictions.shape)
```


```python
# obtain one batch of test images
dataiter = iter(data_loader['val'])
images, labels = dataiter.next()

# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds = torch.max(output, 1)
# prep images for display
images = images.numpy()

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())),
                 color=("green" if preds[idx]==labels[idx] else "red"))
```


```python
# obtain one batch of test images
images, predict = test_x, test_predictions

# convert output probabilities to predicted class
preds = (predict > 0.5).astype('int')

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(Image.open(test_X[idx]))
    if preds[idx] == 0:
        test_predictions[idx] = 1-test_predictions[idx]
    ax.set_title("{} % Accuracy {}".format(str(test_predictions[idx][0]*100), 'dog' if preds[idx]==0 else 'cats'))
```

### Fine tuning


```python
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```


```python
model, history = train_model(model_conv, dataloaders, criterion, optimizer, epochs)
```


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


```python
def preprocess_test(path):
    X = []
    for img_path in path:
        img = image.load_img(img_path, target_size=(224, 224))
        arr = image.img_to_array(img)
        X.append(preprocess_input(arr))
    return np.array(X)

test_x = preprocess_test(test_X)
```


```python
test_predictions = model.predict(test_x)
print (test_predictions.shape)
```


```python
# obtain one batch of test images
images, predict = test_x, test_predictions

# convert output probabilities to predicted class
preds = (predict > 0.5).astype('int')

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(Image.open(test_X[idx]))
    if preds[idx] == 0:
        test_predictions[idx] = 1-test_predictions[idx]
    ax.set_title("{} % Accuracy {}".format(str(test_predictions[idx][0]*100), 'dog' if preds[idx]==0 else 'cats'))
```

<font color='red'>Mr.I-know-everything:</font> Young Padwan, now that you have seen how Transfer Learning works, we will visualize layers in CNN and see what parts of image are they looking at. Visualization layers in CNN plays a crucial role in seeing what is going inside the black box of CNN. Some of the popular visualization techniques include:

- Gradient visualization
- Smooth grad
- CNN filter visualization
- Inverted image representations
- Deep dream
- Class specific image generation

We will implement some of them below

Happy Learning!

## Visualize Activations


```python

```


```python

```


```python
view_layer(model, x, "block1_conv1")
```


```python
view_layer(model, x, "block3_conv1")
```


```python
view_layer(model, x, "block5_conv1")
```


```python

```


```python

```


```python

```


```python

```
