

```
# import shutil 
# shutil.rmtree('data/catsvsdogs')
```

# Transfer Learning

In this notebook, we will go through basics of Transfer Learning and Visualize layers in CNN.

Here we will use [keras](https://keras.io "Keras Homepage").

Hey yo, but what is MLP? what is MNIST? 

Everything is explained in-detail in [blog post](dudeperf3ct.github.io). This is notebook which replicates the result of blog and runs in colab. Enjoy!


#### Run in Colab

You can run this notebook in google colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

# Getting Data


The cats vs dogs dataset isn't available on keras library. You can download it from Kaggle however. Let's see how to do this by using the Kaggle API as it's going to be pretty useful to you if you want to join a competition or use other Kaggle datasets later on.

First, install the Kaggle API by uncommenting the following line and executing it, or by executing it in your terminal.


```
!pip install --upgrade kaggle scikit-plot
```

    Requirement already up-to-date: kaggle in /usr/local/lib/python3.6/dist-packages (1.5.1.1)
    Collecting scikit-plot
      Downloading https://files.pythonhosted.org/packages/7c/47/32520e259340c140a4ad27c1b97050dd3254fdc517b1d59974d47037510e/scikit_plot-0.3.7-py3-none-any.whl
    Requirement already satisfied, skipping upgrade: urllib3<1.23.0,>=1.15 in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.22)
    Requirement already satisfied, skipping upgrade: six>=1.10 in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.11.0)
    Requirement already satisfied, skipping upgrade: certifi in /usr/local/lib/python3.6/dist-packages (from kaggle) (2018.11.29)
    Requirement already satisfied, skipping upgrade: python-dateutil in /usr/local/lib/python3.6/dist-packages (from kaggle) (2.5.3)
    Requirement already satisfied, skipping upgrade: requests in /usr/local/lib/python3.6/dist-packages (from kaggle) (2.18.4)
    Requirement already satisfied, skipping upgrade: tqdm in /usr/local/lib/python3.6/dist-packages (from kaggle) (4.28.1)
    Requirement already satisfied, skipping upgrade: python-slugify in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.2.6)
    Requirement already satisfied, skipping upgrade: scikit-learn>=0.18 in /usr/local/lib/python3.6/dist-packages (from scikit-plot) (0.20.1)
    Requirement already satisfied, skipping upgrade: matplotlib>=1.4.0 in /usr/local/lib/python3.6/dist-packages (from scikit-plot) (2.1.2)
    Requirement already satisfied, skipping upgrade: scipy>=0.9 in /usr/local/lib/python3.6/dist-packages (from scikit-plot) (1.1.0)
    Requirement already satisfied, skipping upgrade: joblib>=0.10 in /usr/local/lib/python3.6/dist-packages (from scikit-plot) (0.13.0)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->kaggle) (3.0.4)
    Requirement already satisfied, skipping upgrade: idna<2.7,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->kaggle) (2.6)
    Requirement already satisfied, skipping upgrade: Unidecode>=0.04.16 in /usr/local/lib/python3.6/dist-packages (from python-slugify->kaggle) (1.0.23)
    Requirement already satisfied, skipping upgrade: numpy>=1.8.2 in /usr/local/lib/python3.6/dist-packages (from scikit-learn>=0.18->scikit-plot) (1.14.6)
    Requirement already satisfied, skipping upgrade: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=1.4.0->scikit-plot) (0.10.0)
    Requirement already satisfied, skipping upgrade: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=1.4.0->scikit-plot) (2.3.0)
    Requirement already satisfied, skipping upgrade: pytz in /usr/local/lib/python3.6/dist-packages (from matplotlib>=1.4.0->scikit-plot) (2018.7)
    Installing collected packages: scikit-plot
    Successfully installed scikit-plot-0.3.7




Then you need to upload your credentials from Kaggle on your instance. Login to kaggle and click on your profile picture on the top left corner, then 'My account'. Scroll down until you find a button named 'Create New API Token' and click on it. This will trigger the download of a file named 'kaggle.json'.

Upload this file to the directory this notebook is running in, by clicking "Upload" on your main Jupyter page, then uncomment and execute the next two commands (or run them in a terminal).



```
#uncomment this once to upload kaggle.json

from google.colab import files
def getLocalFiles():
    _files = files.upload()
    if len(_files) >0:
       for k,v in _files.items():
         open(k,'wb').write(v)
getLocalFiles()
```



     <input type="file" id="files-36a02edc-a146-4857-9a5b-f11458409bed" name="files[]" multiple disabled />
     <output id="result-36a02edc-a146-4857-9a5b-f11458409bed">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script src="/nbextensions/google.colab/files.js"></script> 


    Saving kaggle.json to kaggle.json



```
#uncomment and run this once when you upload kaggle.json

! mkdir -p ~/.kaggle/
! mv kaggle.json ~/.kaggle/
```



```
# This is formatted as code
```

You're all set to download the data from [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) competition. You first need to go to its main page and accept its rules, and run the two cells below (uncomment the shell commands to download and unzip the data). If you get a 403 forbidden error it means you haven't accepted the competition rules yet (you have to go to the competition page, click on Rules tab, and then scroll to the bottom to find the accept button).



```
! kaggle competitions download -c dogs-vs-cats-redux-kernels-edition -p 'data/catsvsdogs'
```

    Warning: Your Kaggle API key is readable by other users on this system! To fix this, you can run 'chmod 600 /root/.kaggle/kaggle.json'
    Downloading test.zip to data/catsvsdogs
     94% 254M/271M [00:02<00:00, 62.8MB/s]
    100% 271M/271M [00:02<00:00, 98.4MB/s]
    Downloading train.zip to data/catsvsdogs
     97% 528M/544M [00:03<00:00, 178MB/s]
    100% 544M/544M [00:03<00:00, 156MB/s]
    Downloading sample_submission.csv to data/catsvsdogs
      0% 0.00/111k [00:00<?, ?B/s]
    100% 111k/111k [00:00<00:00, 110MB/s]



```
path = 'data/catsvsdogs/'

! unzip -q -n {path}/train.zip -d {path}
! unzip -q -n {path}/test.zip -d {path}
```


```
train_path = 'data/catsvsdogs/train/'
val_path = 'data/catsvsdogs/val/'
test_path = 'data/catsvsdogs/test/'
train_cats_dir = f'{train_path}cats/'
train_dogs_dir = f'{train_path}dogs/'
val_cats_dir = f'{val_path}cats/'
val_dogs_dir = f'{val_path}dogs/'
```


```
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import random
from PIL import Image
%matplotlib inline
```


```
print ('Training set images', len(os.listdir(train_path)))
print ('Test set images', len(os.listdir(test_path)))
```

    Training set images 25000
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



```
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

     13%|█▎        | 3234/25000 [00:00<00:00, 32331.24it/s]

    [INFO] Train Folder for dogs and cats created....
    [INFO] Moving train images to dogs and cats folders....


    100%|██████████| 25000/25000 [00:00<00:00, 31706.75it/s]

    [INFO] Moving images from train to cats and dogs complete... 


    




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


```
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

    100%|██████████| 2500/2500 [00:00<00:00, 28134.43it/s]
      0%|          | 0/2500 [00:00<?, ?it/s]

    [INFO] Val Folder created....
    [INFO] Val Folder for dogs and cats created....
    [INFO] Random sample 20% of cats from train to val...
    [INFO] Moving images from train cat to val cat complete...
    [INFO] Random sample 20% of dogs from train to val...


    100%|██████████| 2500/2500 [00:00<00:00, 30603.92it/s]

    [INFO] Moving images from train dog to val dog complete... 


    



```
print ('Training set images', len(os.listdir(train_cats_dir))+len(os.listdir(train_dogs_dir)))
print ('Validation set images', len(os.listdir(val_cats_dir))+len(os.listdir(val_dogs_dir)))
print ('Test set images', len(os.listdir(test_path)))
```

    Training set images 20000
    Validation set images 5000
    Test set images 12500



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

Enough talk, show me the data!


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


![png](transfer_learning_keras_files/transfer_learning_keras_26_0.png)



```
train_x = os.listdir(train_dogs_dir)
# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 10))

for idx in np.arange(10):
    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
    preprocess_img(train_x[idx], ax, 'dog', train_dogs_dir)
    # print out the correct label for each image
```


![png](transfer_learning_keras_files/transfer_learning_keras_27_0.png)


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


![png](transfer_learning_keras_files/transfer_learning_keras_34_0.png)



![png](transfer_learning_keras_files/transfer_learning_keras_34_1.png)



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


![png](transfer_learning_keras_files/transfer_learning_keras_36_0.png)



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


![png](transfer_learning_keras_files/transfer_learning_keras_38_0.png)



```
val_preds = model.predict(val_imgs, batch_size=batch_size)
print (val_preds.shape, val_labels_enc.shape)
```

    (800, 1) (800,)



```
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(val_labels_enc, val_preds.astype('int'), normalize=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f27c9b8c518>




![png](transfer_learning_keras_files/transfer_learning_keras_40_1.png)



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


![png](transfer_learning_keras_files/transfer_learning_keras_46_0.png)



![png](transfer_learning_keras_files/transfer_learning_keras_46_1.png)



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


![png](transfer_learning_keras_files/transfer_learning_keras_48_0.png)



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


![png](transfer_learning_keras_files/transfer_learning_keras_51_0.png)



```
val_preds = model.predict(val_imgs, batch_size=batch_size)
print (val_preds.shape, val_labels_enc.shape)
```

    (800, 1) (800,)



```
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(val_labels_enc, val_preds.astype('int'), normalize=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f27cab16c88>




![png](transfer_learning_keras_files/transfer_learning_keras_53_1.png)



```
from google.colab import files
files.download('finetune.h5')
```

    ----------------------------------------
    Exception happened during processing of request from ('::ffff:127.0.0.1', 41760, 0, 0)
    Traceback (most recent call last):
      File "/usr/lib/python3.6/socketserver.py", line 317, in _handle_request_noblock
        self.process_request(request, client_address)
      File "/usr/lib/python3.6/socketserver.py", line 348, in process_request
        self.finish_request(request, client_address)
      File "/usr/lib/python3.6/socketserver.py", line 361, in finish_request
        self.RequestHandlerClass(request, client_address, self)
      File "/usr/lib/python3.6/socketserver.py", line 721, in __init__
        self.handle()
      File "/usr/lib/python3.6/http/server.py", line 418, in handle
        self.handle_one_request()
      File "/usr/lib/python3.6/http/server.py", line 406, in handle_one_request
        method()
      File "/usr/lib/python3.6/http/server.py", line 639, in do_GET
        self.copyfile(f, self.wfile)
      File "/usr/lib/python3.6/http/server.py", line 800, in copyfile
        shutil.copyfileobj(source, outputfile)
      File "/usr/lib/python3.6/shutil.py", line 82, in copyfileobj
        fdst.write(buf)
      File "/usr/lib/python3.6/socketserver.py", line 800, in write
        self._sock.sendall(b)
    ConnectionResetError: [Errno 104] Connection reset by peer
    ----------------------------------------



```
from google.colab import files
files.download('bottleneck-features.h5')
```

    ----------------------------------------
    Exception happened during processing of request from ('::ffff:127.0.0.1', 35214, 0, 0)
    Traceback (most recent call last):
      File "/usr/lib/python3.6/socketserver.py", line 317, in _handle_request_noblock
        self.process_request(request, client_address)
      File "/usr/lib/python3.6/socketserver.py", line 348, in process_request
        self.finish_request(request, client_address)
      File "/usr/lib/python3.6/socketserver.py", line 361, in finish_request
        self.RequestHandlerClass(request, client_address, self)
      File "/usr/lib/python3.6/socketserver.py", line 721, in __init__
        self.handle()
      File "/usr/lib/python3.6/http/server.py", line 418, in handle
        self.handle_one_request()
      File "/usr/lib/python3.6/http/server.py", line 406, in handle_one_request
        method()
      File "/usr/lib/python3.6/http/server.py", line 639, in do_GET
        self.copyfile(f, self.wfile)
      File "/usr/lib/python3.6/http/server.py", line 800, in copyfile
        shutil.copyfileobj(source, outputfile)
      File "/usr/lib/python3.6/shutil.py", line 82, in copyfileobj
        fdst.write(buf)
      File "/usr/lib/python3.6/socketserver.py", line 800, in write
        self._sock.sendall(b)
    ConnectionResetError: [Errno 104] Connection reset by peer
    ----------------------------------------



```
from keras.models import load_model
model = load_model('finetune.h5')
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
    Trainable params: 26,087,937
    Non-trainable params: 1,735,488
    _________________________________________________________________


    /usr/local/lib/python3.6/dist-packages/keras/engine/saving.py:327: UserWarning: Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
      warnings.warn('Error in loading the saved optimizer '



```

```
