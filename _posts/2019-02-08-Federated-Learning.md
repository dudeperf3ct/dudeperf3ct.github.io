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

- Slow and unreliable Communication

Due to varying upload and download speed across different regions and different countries, the uploads required in federated learning will be very slow compared to traditional distributed machine learning in datacenters where the communications among the nodes is very quick and messages don't get lost (*Imagenet training in 5 mintues*). 



We can 

- Sketeched Updates


- Structured Updates



## Federated Learning Case Study Gboard








# Privacy 

Privacy, the one word which is promised by everyone but delievered by ... (*I will let you complete it*) It's no surprise that with, *With great promises of personalization comes greater responsibility to privacy*.(Thanks Uncle Ben from 2050 Universe)

Apple poster of CES 2019


I will rephrase what Prof. Vivek Wadwa from CMU said about Artificial Intelligence in terms of Privacy,

> Privacy is like


Is the data communicated through federated learning really anonymus and secured? There are primarily two methods, namely secure aggregation and differential privacy to ensure that the data communicated stays anonymized. 



## Secure Aggeration





## Differential Privacy






<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology




---

# Further Reading

Must Read! Florian blog on [Federated Learning](https://florian.github.io/federated-learning/), [Differential Privacy](https://florian.github.io/differential-privacy/) and [Federated Learning on Firefox]()

Must Read! [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf)

[Towards Federated Learning at scale: System Design](https://arxiv.org/pdf/1902.01046.pdf)

[Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/pdf/1610.05492.pdf)

[Google Blog on Federated Learning ](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)

---

# Footnotes and Credits



---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

