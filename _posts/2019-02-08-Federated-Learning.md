---
layout:     post
title:      Differential Privacy and Federated Learning
date:       2019-02-08 12:00:00
summary:    This post will introduce Federated Learning and Differential Privacy.
categories: federated learning
published : false
---

xkcd comic introudcing supervisied learning, unsupervised learning, rl and nlp tasks.

where does this data come from .. invading privacy ... privacy ... hmmm what is that?

# Federated Learning

In the wake of recent events related to privacy invasion through varioys methods of data collections by large corporations, it's about time we think about alternatives about collecting data before more users become aware as to why are they getting such excellent vision, text prediction and recommendation systems. (Hint: by training on their data by invading their privacy, ringing any bells?) One such example is [Detectron](https://florian.github.io/federated-learning/) by Facebook trained on [3 billion images](https://code.facebook.com/posts/1700437286678763/advancing-state-of-the-art-image-recognition-with-deep-learning-on-hashtags/) from Instagram. This is what McMahan and group must have thought (coming from Google itself, a bit ironic isn't it?) and hence, they proposed a learning method through decentralised data in this [remarkable paper](https://arxiv.org/pdf/1602.05629.pdf) through decentralized approach termed as *Federated Learning*.

Amazon graphics from Tamil whatsapp group

Traditional learning algorithms learns on mountain loads of data gathered from many different users which is stored in some central server. Then, distributed learning model is created and trained on mountain loads of user data for months and months. After training, they come back to user promising that they have made their app more intelligent (*without hinting: with help of your data, so mean*).

Federated Learning uses decentralized approach for training the model using the user (*privacy-sensitive*) data. In short, the traditional learning methods had approach to, "brining the data to code", instead of "code to data".

## How it Works?

graphic from florian

In series of rounds(of communication) server selectes K random users(clients) to participate in training. Each selected user downloads the current model from server and performs some number of local updates using its local training data ($$\mathbf{H}_{i}$$); for example it may perform single epoch of minibatch SGD. Then the users upload their model update – that is,the difference between the final parameters after training and the original parameters – and the server averages the contributions before accumulating them into the global model.
 
one graphic from cloudera


Here is simple Federated Averaging algorithm which accumlated the updated from clients into global model.


federated_averaging


In what tasks is federated learning best suited.

1. The task labels don’t require human labelers but are naturally derived from user interaction.
2. The training data is privacy sensitive.
3. The training data is too large to be feasibly collected centrally.


Easy enough?

At first, the training over this decentralized approach looks simple enough and similar to distributed machine learning approaches. But there are some major differences to applications in data centers where the training data is distributed among many machines.

- Non IID 

The data obtained from different users 

- High number of clients

Since, deep learning algorithms are data hungry, applicated using federated learning require a lot of clients. The data from these many users will be far greater than the typically centrally stored data.

- Unbalanced training samples

Each user will have different number of training samples. 

- Slow and unreliable Network Connections

Due to varying upload and download speed across different regions and different countries, the uploads required in federated learning will be very slow compared to traditional distributed machine learning in datacenters where the communications among the nodes is very quick and messages don't get lost (*Imagenet training in 5 mintues*). 

## Compression

Clearly, there is a quite of overhead in communication between client and server given there is unreliable and slow network connections speed. The typical neural networks have millions of parameters nowadays. Sending updates for so many values to a server leads to huge communication costs with a growing number of users and iterations. So, to reduce the uplink communication cost, McMahan and group proposes two methods outlined in [this paper](https://arxiv.org/pdf/1610.05492.pdf). These are the compression techniques which encode updates with fewer bits, as only updates are communicated to server for averaging. 

- **Sketeched Updates**

In this method, each user calculates it's update after training on it's local data, and then before sending the updates to the server, the updates are compressed using a combination of [quantization](https://florian.github.io/probabilistic-quantization/), random rotations and subsampling.

- **Structured Updates**

This second type of compression method restricts the updates to a restricted space such as making each update low-rank with at most rank k or using random mask on updates making it a sparse matrix, and only sending the non-zero entries.

These both methods can be used to reduce the communication overhead and also reduce the size of updates for each round making it reliable even at low upload speeds.

## Applications

- **Smartphones**

Smartphones have revolutionalized the data generation capability with growing number of users hooking on the device each year. With more data, comes more machine learning. Machine learning have provided a lot of cool applications such as Smart Reply, Image Recognition, next word prediction, and many more on smartphones. But this data collection has been heavily relied on private, sensitive user data. Sure, we can make use of synthetic data, but it doesn't capture all the scenarios occuring in real world data. Users (if are aware) are reluctant in sharing such sensitive information which corporations capture(making known or unknown to users) in exchange for smartness. With help of federated learning, the data never leaves the device and model gets trained on large amounts of data. 

- **Healthcare**

This is the field where anonymity plays a very crucial. The consequences of actual and potential privacy violations can be serious. By keeping the training data in the hands of patients or providers, federated learning has the potential to make it possible to collaboratively build models that save lives and generate huge value.


## Federated Learning Case Study Gboard

We will look into one case study of improving suggestions on Gboard done at Google. Authors of the [paper](https://arxiv.org/pdf/1812.02903.pdf) used federated learning(FL) for search query suggestions on Gboard. The goal is to improve query click-through-rate (CTR) by taking suggestions from the baseline model and removing low quality suggestions through the triggering model.

add-image

The use case is to train a model that predicts whether query suggestions are useful, in order to filter out less relevant queries. The training data collected for this model by observing user interactions with the app: when surfacing a query suggestion to a user, a tuple(features; label) is stored in an on-device training cache, a SQLite based database. Here, features is collection of query and context related information and label is user action of {clicked, ignored}. This data is then used for on-device training and evaluation by servers. The model is trained typically at night time when phone is charging, idle and connected to WiFi network. 

The baseline model is traditional server-based machine learning that generates query suggestion candidates by matching the user’s input to an on-device subset ofthe Google Knowledge Graph (KG). It then scores these suggestions using a [Long Short-Term Memory](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#lstm-network) network trained on an offline corpus ofchat data to detect potential query candidates. This LSTM is trained to predict the KG category of a word in a sentence and returns higher scores when the KG category of the query candidate matches the expected category. The highest scoring candidate from the baseline model isselected and displayed as a query suggestion (an impression). The user then either clicks on or ignores the suggestion and users interaction is stored in on-device training cache to be used by FL for training.

The task of the federated trained model is designed to take in the suggested query candidate from the baseline model, and determine if the suggestion should or should not be shown to the user. This FL model is triggering model. The output of model is  a score for a given query, with higher scores meaning greater confidence in the suggestion.


add-image

Here are the steps that are performed in training and updating the global model,

1. The participants in the training are clients(or devices) and FL server which is cloud-based distributed service. Clients annouces to the server that they are ready to run FL task for a given FL population. An FL population is specified by a globally unique name which identifies the learning problem, or application, which is worked upon. An FL task is a specific computation for an FL population, such as training to be performed with given hyperparameters, or evaluation of trained models on local device data. Sever selects some number of clients to run FL task.

2. The server tells the selected devices what computation to run with an FL plan, a data structure that includes a TensorFlow graph and instructions for how to execute it. Once a round is established, the server next sends to each participant the current global model parameters and any other necessary state as an FL checkpoint.

3. Each participant then performs a local computation based on the global state and its local dataset, and sends an update in the form of an FL checkpoint back to the server.

4. The server incorporates these ephemeral updates are aggregated using the Federated Averaging algorithm  into its global state, and the process repeats until convergence. Upon convergence, a trained checkpoint isused to create and deploy a model to clients for inference.

This is one such example demonstrating end-to-end training in FL with decentralized data.

Here is another application of [next word prediction](https://arxiv.org/pdf/1811.03604.pdf) where federation learning can be used. Important result obtained is board is neural language model trained using FL demonstrated better performance than a model trained with traditional server-based collection and training.

# Privacy 

Privacy, the one word which is promised by everyone but delievered by ... (*I will let you complete it*) It's no surprise that with, *With great promises of personalization comes greater responsibility to privacy*.(Thanks Uncle Ben from 2050 Universe)

Apple poster of CES 2019


I will rephrase what Prof. Vivek Wadwa from CMU said about Artificial Intelligence in terms of Privacy,

> Privacy is like


Is the data communicated through federated learning really anonymous and secured? There are primarily two methods, namely secure aggregation and differential privacy to ensure that the data communicated stays anonymized. 



## Secure Aggeration





## Differential Privacy






<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology




---

# Further Reading

Must Read! Florian's blog on [Federated Learning](https://florian.github.io/federated-learning/), [Differential Privacy](https://florian.github.io/differential-privacy/) and [Federated Learning for Firefox](https://florian.github.io/federated-learning-firefox/)

Must Read! [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf)

[Towards Federated Learning at scale: System Design](https://arxiv.org/pdf/1902.01046.pdf)

[Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/pdf/1610.05492.pdf)

[Google Blog on Federated Learning ](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)

[Federated Learning for Mobile Key Prediction](https://arxiv.org/pdf/1811.03604.pdf)

[Learning differentially private language models without losing accuracy](https://arxiv.org/pdf/1710.06963)

[Practical Secure Aggregation for Privacy-Preserving Machine Learning](http://delivery.acm.org/10.1145/3140000/3133982/p1175-bonawitz.pdf)

[Deep Learning with Differential Privacy](https://arxiv.org/pdf/1607.00133.pdf)

---

# Footnotes and Credits

[Federated Averaging Algorithm](https://arxiv.org/pdf/1602.05629.pdf)


---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

