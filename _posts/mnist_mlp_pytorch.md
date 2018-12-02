
# MLP

In this notebook, we will go through basics of MLP using MNIST dataset. We will implement this using two popular deep learning frameworks `Keras` and `PyTorch`. 

Here we will use [PyTorch](https://pytorch.org/tutorials/ "PyTorch's Tutorial").

Hey yo, but what is MLP? what is MNIST? 

Well sit tight and buckle up. I will go through everything in-detail.


```python

```


```python
# load all the required libraries
import time
import copy
import numpy as np                                    # package for computing 
from torch.utils.data.sampler import SubsetRandomSampler # split dataset
import torch                                          # import torch
from torchvision import datasets                      # import dataset from torch
import torchvision.transforms as transforms           # apply transformations to data
import torch.nn as nn                                 # neural network modules
import torch.nn.functional as F                       # functional api
from torch import optim                               # optimizers

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt             # matplotlib library for plotting

# display plots inline (in notebook itself)
%matplotlib inline                          
```

## MNIST Dataset


 
The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples each of size 28 x 28 pixels. The digits have been size-normalized and centered in a fixed-size image.

It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting. 

Here is one example from dataset





```python
# load mnist data

# number of subprocesses to use for data loading
num_workers = 0

# how many samples per batch to load
batch_size = 20

# percentage of training set to use as validation
val_size = 0.2

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

# obtain training indices that will be used for validation
num_test = len(test_data)
indices = list(range(num_test))
np.random.shuffle(indices)
split = int(np.floor(val_size * num_test))
test_idx, val_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
test_sampler = SubsetRandomSampler(test_idx)
val_sampler = SubsetRandomSampler(val_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                         sampler=val_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          sampler=test_sampler, num_workers=num_workers)

print ('Training samples:', len(train_loader.dataset))
print ('Validation samples:', len(val_loader.dataset))
print ('Testing samples:',len(test_loader.dataset))

# Create training and validation dataloaders
dataloaders_dict = {'train': train_loader, 
                    'val': val_loader}

```

    Training samples: 60000
    Validation samples: 10000
    Testing samples: 10000


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



## Visualization of data

Enough talk, show me the data!


```python
import matplotlib.pyplot as plt
%matplotlib inline
    
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    # print out the correct label for each image
    # .item() gets the value contained in a Tensor
    ax.set_title(str(labels[idx].item()))
```


![png](mnist_mlp_pytorch_files/mnist_mlp_pytorch_7_0.png)



```python
img = np.squeeze(images[1])

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


![png](mnist_mlp_pytorch_files/mnist_mlp_pytorch_8_0.png)


# Introduction to MLP



MLP is multi-layer percepton. Perceptron is a single layer neural network and a multi-layer perceptron is called Neural Networks.  
We have seen the dataset, which consist of [0-9] numbers and images of size 28 x 28 pixels of values in range [0-1] . 

Now, <font color='green'>Mr.I-know-nothing</font> being too lazy to find which number is what asks for <font color='red'>Mr.I-know-everything</font> apprenticeship to create a Machine Learning Model such that if we pass a grayscale image of size 28 x 28 pixels to the model, it outputs a correct label corresponding to that image. 

<font color='blue'> A long time ago in a galaxy far, far away.... </font>

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

<font color='green'>Mr.I-know-nothing:</font> Thank you Master, I follow.

<font color='red'>Mr.I-know-everything:</font> Now you are in for a treat. As you have learn about what different terms and functions are used to train a neural network. We will dive-in implementation using `PyTorch saber`. Here backpropogation is already implemented i.e. you only need to design forward pass and loss(or error) function, the framework takes care to backward pass. 

## Sequential API


```python
# [0-9] unique labels
num_classes = 10
epochs = 5
```


```python
## Define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        D_in = 784
        H = 16
        D_out = num_classes
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
            torch.nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # 1 hidden layer, with relu activation function
        x = self.classifier(x)
        return x

# initialize the NN
model = Net()
model = model.to(device)
print(model)
```

    Net(
      (classifier): Sequential(
        (0): Linear(in_features=784, out_features=16, bias=True)
        (1): ReLU()
        (2): Linear(in_features=16, out_features=10, bias=True)
        (3): LogSoftmax()
      )
    )



```python
# specify loss function
criterion = nn.NLLLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```


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
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

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
model, history = train_model(model, dataloaders_dict, criterion, optimizer, epochs)
```

    Epoch 0/4
    ----------
    train Loss: 0.4130 Acc: 0.8860
    60000
    val Loss: 0.2651 Acc: 0.9215
    10000
    Epoch 1/4
    ----------
    train Loss: 0.2437 Acc: 0.9299
    60000
    val Loss: 0.2316 Acc: 0.9275
    10000
    Epoch 2/4
    ----------
    train Loss: 0.2067 Acc: 0.9402
    60000
    val Loss: 0.2121 Acc: 0.9330
    10000
    Epoch 3/4
    ----------
    train Loss: 0.1851 Acc: 0.9464
    60000
    val Loss: 0.1969 Acc: 0.9395
    10000
    Epoch 4/4
    ----------
    train Loss: 0.1699 Acc: 0.9512
    60000
    val Loss: 0.1891 Acc: 0.9435
    10000
    Training complete in 0m 50s
    Best val Acc: 0.943500



```python
plt.plot(history['train_acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history['train_loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
```


![png](mnist_mlp_pytorch_files/mnist_mlp_pytorch_17_0.png)



![png](mnist_mlp_pytorch_files/mnist_mlp_pytorch_17_1.png)



```python
# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for *evaluation*

for data, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
```

    Test Loss: 0.150191
    
    Test Accuracy of     0: 97% (760/783)
    Test Accuracy of     1: 98% (906/919)
    Test Accuracy of     2: 92% (762/824)
    Test Accuracy of     3: 89% (721/804)
    Test Accuracy of     4: 94% (751/793)
    Test Accuracy of     5: 97% (698/718)
    Test Accuracy of     6: 93% (730/779)
    Test Accuracy of     7: 94% (765/806)
    Test Accuracy of     8: 90% (706/777)
    Test Accuracy of     9: 92% (740/797)
    
    Test Accuracy (Overall): 94% (7539/8000)



```python
# obtain one batch of test images
dataiter = iter(test_loader)
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


![png](mnist_mlp_pytorch_files/mnist_mlp_pytorch_19_0.png)


## Functional API


```python
# [0-9] unique labels
num_classes = 10
epochs = 5
```


```python
## Define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_hidden1 = 16
        # linear layer (784 -> 1 hidden node)
        self.fc1 = nn.Linear(28 * 28, self.num_hidden1)
        self.output = nn.Linear(self.num_hidden1, num_classes)
        
    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # 1 hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.output(x), dim=1)
        return x

# initialize the NN
model = Net()
model = model.to(device)
print(model)
```

    Net(
      (fc1): Linear(in_features=784, out_features=16, bias=True)
      (output): Linear(in_features=16, out_features=10, bias=True)
    )



```python
# specify loss function
criterion = nn.NLLLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```


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
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

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
model, history = train_model(model, dataloaders_dict, criterion, optimizer, epochs)
```

    Epoch 0/4
    ----------
    train Loss: 0.4015 Acc: 0.8915
    60000
    val Loss: 0.2860 Acc: 0.9155
    10000
    Epoch 1/4
    ----------
    train Loss: 0.2460 Acc: 0.9302
    60000
    val Loss: 0.2658 Acc: 0.9195
    10000
    Epoch 2/4
    ----------
    train Loss: 0.2112 Acc: 0.9394
    60000
    val Loss: 0.2480 Acc: 0.9230
    10000
    Epoch 3/4
    ----------
    train Loss: 0.1895 Acc: 0.9458
    60000
    val Loss: 0.2399 Acc: 0.9280
    10000
    Epoch 4/4
    ----------
    train Loss: 0.1748 Acc: 0.9497
    60000
    val Loss: 0.2310 Acc: 0.9290
    10000
    Training complete in 0m 52s
    Best val Acc: 0.929000



```python
plt.plot(history['train_acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history['train_loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
```


![png](mnist_mlp_pytorch_files/mnist_mlp_pytorch_26_0.png)



![png](mnist_mlp_pytorch_files/mnist_mlp_pytorch_26_1.png)



```python
# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for *evaluation*

for data, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
```

    Test Loss: 0.174892
    
    Test Accuracy of     0: 96% (757/783)
    Test Accuracy of     1: 99% (911/919)
    Test Accuracy of     2: 90% (748/824)
    Test Accuracy of     3: 89% (719/804)
    Test Accuracy of     4: 94% (752/793)
    Test Accuracy of     5: 97% (697/718)
    Test Accuracy of     6: 94% (735/779)
    Test Accuracy of     7: 91% (741/806)
    Test Accuracy of     8: 84% (654/777)
    Test Accuracy of     9: 94% (754/797)
    
    Test Accuracy (Overall): 93% (7468/8000)



```python
# obtain one batch of test images
dataiter = iter(test_loader)
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


![png](mnist_mlp_pytorch_files/mnist_mlp_pytorch_28_0.png)


<font color='red'>Mr.I-know-everything:</font> Young Padwan, now you have the same power as me to train an MLP. Now knock yourself and experiement with different number of layers. Also, watch for training and validation loss as hint if model is moving in right direction. There you will come across `overfitting` and `underfiting`. So, be sure to watch them and we will discuss about them in detail in next time where you will learn about `Force of CNN` and how they can further give us best model (Yes, better than MLP). Until next time, try different architectures and keep researching.

<font color='green'>Mr.I-know-nothing:</font> Thank you Master.

Happy Learning!


```python

```
