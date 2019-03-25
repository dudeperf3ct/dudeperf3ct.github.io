---
layout:     post
title:      Power of Transfer Learning in NLP
date:       2019-02-22 12:00:00
summary:    This post will provide a brief introduction 
categories: nlp transfer learning
published : false
---


# Transfer Learning in NLP

In this notebook, .

> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/lstm_and_gru/lstm_and_gru_keras.ipynb), [PyTorch](https://github.com/dudeperf3ct/DL_notebooks/blob/master/lstm_and_gru/lstm_and_gru_pytorch.ipynb), [Flair](https://github.com/dudeperf3ct/DL_notebooks/blob/master/lstm_and_gru/lstm_and_gru_flair.ipynb) and [fastai](https://github.com/dudeperf3ct/DL_notebooks/blob/master/tl_nlp/lstm_and_gru_fastai.ipynb).

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.


Feel free to jump anywhere,

- [NLP Tasks and Dataset](#nlp-tasks-and-datasets)
  - [Sentiment Analysis](#sentiment-analysis)
  - [POS](#pos)
  - [NER](#ner)
  - [Textual Entailment](#textual-entailment)
  - [Coreference resolution](#coreference-resolution)
  - [Question Answering](#question-answering)
- [CoVe](#cove)
- [ELMo](#elmo)
- [ULMFiT](#ulmfit)
- [GPT](#gpt)
- [BERT](#bert)
- [GPT-2](#gpt-2)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)

## NLP Tasks and Datasets


### Sentiment analysis

**Task**

Sentiment analysis is task of classifying polarity of given text.

**Dataset**

- [IMDb](https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf)
- [Stanford Sentiment Treebank (SST)](https://nlp.stanford.edu/sentiment/index.html)
- [Yelp Review dataset](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)

**Current SoTA** : [Sentiment Analysis](http://nlpprogress.com/english/sentiment_analysis.html)

**Sample Example**

- Input 

Sentence: Avengers Endgame is the best movie. Kudos Russo brothers.

- Output 

Positive (100% accuracy) (sentiment)

### POS

**Task**

A Part-Of-Speech Tagger (POS Tagger) is a piece of software that reads text in some language and assigns parts of speech to each word (and other token), such as noun, verb, adjective, etc.

**Dataset**

- [Penn Treebank](https://github.com/tomsercu/lstm/tree/master/data)
- [Social Media](https://aclanthology.coli.uni-saarland.de/papers/D11-1141/d11-1141)
- [Universal Dependencies (UD)](http://universaldependencies.org/) 

**Current SoTA** : [Part-of-speech tagging](http://nlpprogress.com/english/part-of-speech_tagging.html)


**Sample Example**

- Input 

Sentence: Apple is looking at buying U.K. startup for $1 billion

- Output

<p align="center">
<img src='/images/tl_nlp/pos_ex.png' width="40%"/>
<img src='/images/tl_nlp/pos_ex_1.png' width="40%"/>
</p>

**Note**: Output obtained from [spaCy POS Tagging](https://spacy.io/usage/linguistic-features#pos-tagging). Try now!


### NER

**Task**

Named Entity Recognition (NER) labels sequences of words in a text which are the names of things, such as person and company names, or gene and protein names. 

**Dataset**

- [CoNLL 2003 NER task](http://www.aclweb.org/anthology/W03-0419.pdf)
- [WNUT 2017 Emerging Entities task](http://aclweb.org/anthology/W17-4418)
- [Ontonotes corpus v5](https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf) 

**Current SoTA** : [Named entity recognition](http://nlpprogress.com/english/named_entity_recognition.html)

**Sample Example**

- Input 

Sentence: Apple is looking at buying U.K. startup for $1 billion

- Output

<p align="center">
<img src='/images/tl_nlp/ner_ex.png' width="40%"/>
<img src='/images/tl_nlp/ner_ex_1.png' width="40%"/>
</p>


**Note**: Output obtained from [spaCy Named Entities](https://spacy.io/usage/linguistic-features#named-entities). Try now!

Also, here is a live demo from Allennlp for [Named Entity Recognition](https://demo.allennlp.org/named-entity-recognition).


### Textual Entailment

**Task**

Textual Entailment (TE) also known as Natural language inference (NLI) takes a pair of sentences and predicts whether the facts in the first necessarily imply the facts in the second one or task of determining whether a “hypothesis” is true (entailment), false (contradiction), or undetermined (neutral) given a “premise”.

**Dataset**

- [SNLI](https://nlp.stanford.edu/projects/snli/)
- [MultiNLI](Multi-Genre Natural Language Inference (MultiNLI) corpus)
- [SciTail](http://ai2-website.s3.amazonaws.com/publications/scitail-aaai-2018_cameraready.pdf) 

**Current SoTA** : [Natural language inference](http://nlpprogress.com/english/natural_language_inference.html)


**Sample Example**

- Input 

Premise : If you help the needy, God will reward you.

Hypothesis : Giving money to the poor has good consequences.

- Output

<p align="center">
<img src='/images/tl_nlp/nli_ex.png' width="90%"/>
</p>

**Note**: Here is a live demo from Allennlp for [Textual Entailment](https://demo.allennlp.org/textual-entailment).


### Coreference resolution


**Task**

Coreference resolution is the task of finding all expressions that refer to the same entity in a text.

For e.g. The trophy would not fit in the brown suitcase because it was too big (small). What was too big (small)?

Answer 0: the trophy

Answer 1: the suitcase 

**Dataset**

- [CoNLL-2012 shared task](http://www.aclweb.org/anthology/W12-4501)

**Current SoTA** : [Coreference resolution](http://nlpprogress.com/english/coreference_resolution.html)

**Sample Example**

- Input 

Sentence: The trophy would not fit in the brown suitcase because it was too big. What was too big? the trophy or the suitcase?

- Output 

<p align="center">
<img src='/images/tl_nlp/coref_ex.png' width="90%"/>
</p>

**Note**: Here is a live demo from huggingface for [Coreference resolution](https://huggingface.co/coref/?text=The%20trophy%20would%20not%20fit%20in%20the%20brown%20suitcase%20because%20it%20was%20too%20big.%20What%20was%20too%20big%3F%20the%20trophy%20or%20the%20suitcase%3F). Also check [Winograd Challenge](http://commonsensereasoning.org/winograd.html).

Here is a live demo from Allennlp for [Coreference resolution](https://demo.allennlp.org/coreference-resolution).


### Question Answering

**Task**

Reading comprehension or Question Answering is the task of answering questions about a passage of text to show that the system understands the passage

**Dataset**

- [AI2 Reasoning Challenge (ARC)](http://ai2-website.s3.amazonaws.com/publications/AI2ReasoningChallenge2018.pdf)
- [ShARC](https://arxiv.org/abs/1809.01494)
- [CliCR dataset](http://aclweb.org/anthology/N18-1140) 
- [CNN / Daily Mail dataset](https://arxiv.org/abs/1506.03340)
- [CoQA](https://arxiv.org/abs/1808.07042) 
- [HotpotQA](https://hotpotqa.github.io/)
- [MS MARCO](http://www.msmarco.org/leaders.aspx) 
- [MultiRC](http://cogcomp.org/multirc/)
- [NewsQA dataset](https://arxiv.org/pdf/1611.09830.pdf) 
- Stanford Question Answering Dataset (SQuAD) [SQUAD 1.0](https://arxiv.org/abs/1606.05250) and [SQUAD 2.0](https://arxiv.org/abs/1806.03822)
and many more!

*Check [nlpprogress](http://nlpprogress.com/english/question_answering.html) for complete list.*

**Current SoTA** : [Question Answering](http://nlpprogress.com/english/question_answering.html)

**Sample Example**

<p align="center">
<img src='/images/tl_nlp/gpt_2_qa.png' width="80%"/>
</p>


**Note**: Here is a live demo from Allennlp for [QA](https://demo.allennlp.org/reading-comprehension).

---

There are many more challenges and [nlpprogess](http://nlpprogress.com/) provides a great overview of challenges and current SOTA for each challenge. Be sure to check it out!

--

# Transfer Learning in NLP

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' /> 
</p>


<span class='red'>I-know-everything:</span> Today the topic of interest is very interesting. It's Transfer Learning in NLP. Can we transfer the knowledge learned about the language and fine-tune it to task at hand. It's the similar concept we saw in [Power of Transfer Learning](https://dudeperf3ct.github.io/transfer/learning/catsvsdogs/2018/11/20/Power-of-Transfer-Learning/) for Computer Vision.


<span class='green'>I-know-nothing:</span> Will we be using same embedding models which we learned in previous posts?


<span class='red'>I-know-everything:</span> Well, there's a catch and to answer your question no. We will not be using traditional embedding models.

The [embedding models](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#embeddings) which we disscused earlier like [word2vec](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#word2vec), [GLoVe](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#glove) and [fastText](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#fasttext) are fantastic in capturing meaning of individual words and their relationships by leveraging large datasets. These model generate word vectors of n-dimension which is used by neural network as starting point of training. The word vectors can be initialized to lists of random numbers before a model is trained for a specific task, or initialized with word vectors obtained from above embedding models.

Here is one such relationship learned through embeddings,

<p align="center">
<img src='/images/tl_nlp/word2vec.png' width="50%"/>
</p>

*How amazingly word2vec learns the captials and relation with the countries?* Just through simple arithmetic algebra, a + b - c gives the correct answer.

In above embedding models, a word is assigned the same vector representation no matter where it appears and how it's used, because word embeddings rely on just a look-up table. In other word, they ignore *polysemy* — a concept that words can have multiple meanings. To take this point home, let's consider a example, The way Messi *plays* football, can only be par with the greatest Broadway *plays*. Notice the word *plays* in the sentence, the first plays is related to playing while the second plays is more related to drama. The traditional embedding models will assign the same vector for both words when in turn we need embedding that also takes into consideration the context in which the word is used. Those are the embeddings we will learn about in following approaches and how can we achieve such *context-conscious* embeddings.

The basic idea of following approaches which we will look into will be to learn representation (depending on context) instead fixed emebedding of each word by training a deep language model and use the representation learned by the language model in downstream tasks.


## CoVe

In NLP tasks, context matters. That is, understanding context is very essential to all NLP tasks as words rarely appear in isolation and also helps in general sense of language understanding tasks. One such example is in Question Answering where understanding of how words in question shift the importance of words in document or in Summarization where model needs to understand which words capture the context clearly to summarize succinctly. The ability to share a common representation of words in the context of sentences that include them could further improve transfer learning in NLP. This is where CoVe comes into play which transfers information from large amounts of unlabeled training data in the form of word vectors using encoder to contextualize word vector has shown to improve performance over random word vector initialization on a variety of downstream tasks e.g. POS, NER and QA. 

### How it Works?

CoVe is Contextual Word Vectors, type of word embedding learned by encoder in an attentional seq-seq machine translational model.
The team at Salesforce explained CoVe in best way on their [research blog](https://blog.einstein.ai/learned-in-translation-contextualized-word-vectors/), also outlined in [their paper](https://arxiv.org/pdf/1708.00107.pdf). We will look at a special case example of Machine Translation from English (source language) to German (target language).

<p align="center">
<img src='/images/tl_nlp/cove.png' width="70%"/>
</p>


- **Encoder**

A neural network [BiLSTM](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#bidirectional-lstm-network) takes word vectors as input and outputs a new vector called hidden vector. This process is often referred to as encoding the sequence, and the neural network that does the encoding is referred to as an encoder. BiLSTM (forward and backward LSTM) is used to incorporate information from words that appear later in the sequence.

<p align="center">
<img src='/images/tl_nlp/encoder_cove.svg' width="70%"/>
</p>

CoVe uses two BiLSTM layers as encoder, first BiLSTM processes its entire sequence before passing outputs to the second. Let $$w^{x}$$ = [$$w^{x}_{1}, w^{x}_{2}, ..., w^{x}_{n}$$] sequence of words in source language, then the output hidden vector h or CoVe vector,

$$
\begin{aligned}
\textbf{Encoder} : CoVe(w) & = BiLSTM(GloVe(w^{x})) \\
h = [h_{1}, h_{2}, .. h_{n}]  & = BiLSTM(GloVe(w^{x})) \\
h_{t} &  = [\overset{\leftarrow}{h_{t}}; \overset{\rightarrow}{h_{t}}] \\
\overset{\leftarrow}{h_{t}} &  = LSTM(GloVe(w^{x_t}), \overset{\leftarrow}{h_{t-1}}) \\
\overset{\rightarrow}{h_{t}} & = LSTM(GloVe(w^{x_t}), \overset{\rightarrow}{h_{t-1}}) \\
\end{aligned}
$$

The pretrained vectors obtained from embedding captured some interesting relationships, similar results are obtained from hidden vectors (h). In our case of Machine Translation, inputs are Glove embeddings of English words of input sentence (GloVe($$w^{x}$$)) and output are it's hidden vectors(h). After training we call this pretrained LSTM an MT-LSTM (Machine Translation) and can serve as pretrained model to generate hidden vectors for new sentences. When using these machine translation hidden vectors as inputs to another NLP model, we refer to them as context vectors (CoVe).

- **Decoder**

Encoder produces hidden vector for English sentences given input different English sentences. Another neural network called decoder references those hidden vectors to generate the German sentence. The decoder LSTMs is initialized from the final states of the encoder, reads in a special German word vector to start, and generates a decoder state vector.

<p align="center">
<img src='/images/tl_nlp/decoder_cove.svg' width="70%"/>
</p>

Tbe decoder takes in input randomly intialized embedding for target words $$w^{z}$$ = [$$w^{z}_{1}, w^{z}_{2}, ..., w^{z}_{n}$$] , context-adjusted state generated by attention mechanism and previous hidden state of LSTM. Similar to encoder, decoder uses two layer LSTM (unidirectional) to create decoder state from input word vectors.


$$
\begin{aligned}
\textbf{Decoder hidden state} : s_{t} = LSTM([w^{z}_{t-1}; \tilde{h}_{t-1}], s_{t-1})
\end{aligned}
$$

- **Attention**

Attention mechanism is one the interesting mechanism in NLP and we will look into more depth in next post. The attention mechanism looks back at hidden vectors in order to decide which part of English sentence to translate next. It uses the state vector to determine how important each hidden vector is, and then it produces a new vector, which we will call the context-adjusted state, to record its observation.

<p align="center">
<img src='/images/tl_nlp/attention_cove.svg' width="70%"/>
</p>

Attention mechanism uses hidden vectors generated by encoder and decoder state by decoder to produce context-adjusted state. It sort of plays a role of deciding which context words play important role in translation and focusing on them rather than whole sentence.

$$
\begin{aligned}
\textbf{Attention Weights} : \alpha_{t} & = softmax(H(W_{1}s_{t} + b_{1}))\\ 
\textbf{Context-adjusted Weights} : \tilde{h}_{t} & = tanh(W_{2}[H^{T} \alpha_{t}; s_{t}] + b_{2})\\
\end{aligned}
$$

Here H is is a stack of hidden states {h} along the time dimension.

- **Generation**

The generator (not a sperate layer, it's a decoder but step is generation because it generates output sentence) then looks at the context-adjusted state to determine which German word to output, and the context-adjusted state is passed back to the decoder so that it has an accurate sense of what it has already translated. The decoder repeats this process until it is done translating. This is a standard attentional encoder-decoder architecture for learning sequence to sequence tasks like machine translation.

-generator_cove
<p align="center">
<img src='/images/tl_nlp/generator_cove.svg' width="70%"/>
</p>


The generator uses context-adjusted state from attention mechanism to produce output German word. Attention mechanism takes in input all hidden vectors and first decoder state to produce context-adjusted state which will be used as input to decoder to produce another decoder state. This decoder state along with all hidden vectors will again be used as input to attention mechanism to generate another context-adjusted state which will be input to third decoder and so on.

$$
\begin{aligned}
\textbf{Generator Output} : p(y_{t} \mid H,y_{1},y_{2}, …,y_{t-1}) & = softmax(W_{out}\tilde{h}_{t} + b_{out}))
\end{aligned}
$$

Here $$p(y_{t} \mid H,y_{1},y_{2}, …,y_{t-1})$$ is a probability distribution over output words.

### TL;DR

- Use the traditional encoder-decoder architecture used in seq-to-seq learning, to learn the context of words by giving input GLoVe embedding of words in sentence to encoder and two stacked BiLSTM layers generate output is hidden vector or context vectors.
- We looked at one specific example of MT, where encoder was used to generate context vectors, and this context vectors along with attention mechanism (which gives context-adjusted state as output) to give target langauge output sentence using decoder.

### Results

Hmm, that seems simple process but what about results? Was is it SOTA breaker?

<p align="center">
<img src='/images/tl_nlp/cove_results.png' width="70%"/>
</p>

Here CoVe+GLove means that we take the GloVe sequence, run it through a pretrained MT-LSTM to get CoVe sequence, and we append each vector in the CoVe sequence with the corresponding vector in the GloVe sequence.

<p align="center">
<img src='/images/tl_nlp/cove_results_1.png' width="70%"/>
</p>

SOTA in 3 out of 7 tasks, well that's  a good start with using CoVe pretrained vectors.

<p align="center">
<img src='/images/tl_nlp/cove_results_overall.png' width="70%"/>
</p>

### What this means?

Replacing the good ol' GloVe, Word2vec and fastText with CoVe seems to do a good job at the tasks where context matters. Training a custom pretrained CoVe model is also simple. Just take any unlabelled data corresponding to task at hand (e.g. Amazon Review for SST or IMDB 50,000 unlabelled reviews for IMDb sentiment analysis task) pass it through encoder (MT-LSTM) to generate CoVe word vector in supervised fashion and we can use that CoVe pretrained vector along with GloVe vector as initial embedding model and use that train for specific task like sentiment analysis, Question Answering, Machine Translation, etc. The more data we use to train the MT-LSTM, the more pronounced the improvement, which seems to be complementary to improvements that come from using other forms of pretrained vector representations.

<p align="center">
<img src='/images/tl_nlp/general_cove.png' width="70%"/>
</p>

Here there is disadvantage of using only avaliable data for generating pretrained CoVe embedding using supervised training of encoder-decoder architecture. (*no large unsupervisied dataset which are everywhere, supervised learning requires labels too*)

## ELMo

Hi, my name is ELMo and I will overcome the limitation of CoVe by generating contextual embeddings in an unsupervised fashion.

<p align="center">
<img src='/images/tl_nlp/elmo.jpeg' width="50%"/>
</p>

ELMo stands for Embeddings from Language Models. ELMo is a word representation technique proposed by [AllenNLP](https://arxiv.org/pdf/1802.05365.pdf)

### How it Works?

ELMo word representations are function of entire input sentence and are computed on top of two biLM with character convolutions.

- **Bidirectional Language Model**

A language model is an NLP model which learns to predict the next word in a sentence. For instance, if your mobile phone keyboard guesses what word you are going to want to type next, then it’s using a language model. The reason this is important is because for a language model to be really good at guessing what you’ll say next, it needs a lot of world knowledge (e.g. “I ate a hot” → “dog”, “It is very hot” → “weather”), and a deep understanding of grammar, semantics, and other elements of natural language.

<p align="center">
<img src='/images/tl_nlp/elmo_bilm.png' width="70%"/>
</p>

Given a sequence of N tokens, ($$t_{1}, t_{2}, ..., t_{N}$$) forward language model(LM) computes the probability of sequence by modelling the probability of token $$t_{k}$$ given history ($$t_{1}, t_{2}, ..., t_{k-1}$$):

$$
\begin{aligned}
p(t_{1}, t_{2}, ..., t_{N}) = \prod_{k=1}^{N}p(t_{k} \mid t_{1}, t_{2}, ..., t_{k-1})
\end{aligned}
$$

Given a sequence of N tokens, ($$t_{1}, t_{2}, ..., t_{N}$$) backward language model(LM) computes the probability of sequence by modelling the probability of predicting previous token $$t_{k}$$ given future context ($$t_{k+1}, t_{k+2}, ..., t_{N}$$):

$$
\begin{aligned}
p(t_{1}, t_{2}, ..., t_{N}) = \prod_{k=1}^{N}p(t_{k} \mid t_{k+1}, t_{k+2}, ..., t_{N})
\end{aligned}
$$

A bidirectional language model consists of forward LM and backward LM and combines both a forward and backward LM. This model is trained to minimize the negative log likelihood (= maximize the log likelihood for true words) of forward and backward directions:

$$
\begin{aligned}
\mathcal{L}_{LM} = \sum_{k=1}^{N}(log (p(t_{k} \mid t_{1}, t_{2}, ..., t_{k-1}); \Theta_{x},  \overset{\rightarrow}\Theta_\text{LSTM}, \Theta_{s}) \\ + log (p(t_{k} \mid t_{k+1}, t_{k+2}, ..., t_{N}); \Theta_{x},  \overset{\leftarrow}\Theta_\text{LSTM}, \Theta_{s}))
\end{aligned}
$$

Here $$\Theta_{x}$$ and $$\Theta_{s}$$ are embedding layers and softmax layers. Overall, this formulation is similar to the approach of CoVe, with the exception that we share some weights between directions instead of using completely independent parameters. The internal states of forward pass at a certain word reflect the word itself and what has happened before that word, whereas similar can be concluded for backward pass where word itself and what has happened after that word gets reflected. These two passes are concatenated to get intermediate word vector of that word. Therefore, this intermediate word vector at that word is still the representation of what the word means, but it "knows" what is happening (i.e. captures the essence or context) in the rest of the sentence and how the word is used.

<p align="center">
<img src='/images/tl_nlp/elmo_bilstm.png' width="70%"/>
</p>

ELMo uses two layer biLM where each biLM layer consists of one forward pass and one backward pass that scans the sentence in both directions. ELMo is a task specific combination of the intermediate layer representations in the biLM. For each token $$t_{k}$$, a L-layer biLM computes a set of L+1 representations:

$$
\begin{aligned}
\mathcal{R}_{k} & = \{x_{k}^{LM}, \overset{\rightarrow}h_{k,j}^{LM} ,\overset{\leftarrow}h_{k,j}^{LM} \mid j=1,2,...,L\} \\
& = \{h_{k, j}^{LM} \mid j = 0,1,2,...,L\}
\end{aligned}
$$

where $$h_{k, 0}^{LM}$$ is embedding layer $$h_{k, j}^{LM} = [\overset{\rightarrow}h_{k,j}^{LM} ; \overset{\leftarrow}h_{k,j}^{LM}]$$, for each biLSTM layer.

For inclusion in a downstream model, ELMo collapses all layer in $$\mathcal{R}$$ into a single vector,

$$
\begin{aligned}
\text{ELMo}_{k}^{task} & = E[\mathcal{R}_{k}; \Theta^{task}]\\
& = \gamma^{task} \sum_{j=0}^{L}s_{j}^{task}h_{k, j}^{LM}
\end{aligned}
$$

where $$s^{task}$$ are softmax-normalized weights and the scalar parameter $$\gamma^{task}$$ allows the task model to scale the entire ELMo vector.

Finally, ELMo uses character CNN (convolutional neural network) for computing those raw word embeddings that get fed into the first layer of the biLM. The input to the biLM is computed purely from characters (and combinations of characters) within a word, without relying on some form of lookup tables like we had in case of [word2vec](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#word2vec) and [glove](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#glove). This type of character n-gram were seen in [fastText](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#fasttext) embeddings and are very much known for their way of handling OOV (out of vocabulary) words. Thus, ELMo embeddings can handle OOV in efficient manner.

Study of "what information is captured by biLM representations" section of [paper](https://arxiv.org/pdf/1802.05365.pdf) indicate that syntactic information is better represented at lower layers while semantic information is captured by higher layers. Because different layers tend to carry different type of information, stacking them together helps.

[Masato Hagiwara](http://www.realworldnlpbook.com/blog/improving-sentiment-analyzer-using-elmo.html) points out difference between biLM and biLSTM clearly,
 
<span class='red'>A word of caution: the biLM used by ELMo is different from biLSTM although they are very similar. biLM is just a concatenation of two LMs, one forward and one backward. biLSTM, on the other hand, is something more than just a concatenation of two spearate LSTMs. The main difference is that in biLSTM, internal states from both directions are concatenated before they are fed to the next layer, while in biLM, internal states are just concatenated from two independently-trained LMs.</span>

### TL;DR

- Different words carry different meaning depending on context and so their embeddings should also take context in account.
- ELMo trains a bidirectional LM, and extract the hidden state of each layer for the input sequence of words.
- Then, compute a weighted sum of those hidden states to obtain an embedding for each word. The weight of each hidden state is task-dependent and is learned.
- This learned ELMo embedding in used in specific task for which embedding is obtained.

### Results

Well, ELMo certainly outperforms CoVe and emerges as new SOTA at all the 6 tasks with relative error reductions ranging from 6 - 20%. 

<p align="center">
<img src='/images/tl_nlp/elmo_result.png' width="70%"/>
</p>

### What this means?

Here is one the results from context embedding of biLM.

<p align="center">
<img src='/images/tl_nlp/bilm_example.png' width="70%"/>
</p>

Notice how biLM s able to disambiguate both the part of speech and word sense in the source sentence of word "play" than glove counterpart which has fixed neighbours no matter the context.

ELMo improves task performance over word vectors as the biLM’s contextual representations encodes information generally useful for NLP tasks that is not captured in word vectors.

Once pretrained, the biLM can compute representations for any task. In some cases, fine tuning the biLM on domain specific data leads to significant drops in perplexity and an increase in down-stream task performance. 

Given a pretrained LM and a supervised architecture for a target NLP task, it is a simple process to use the biLM to improve the task model. We simply run the biLM and record all of the layer representations for each word. Then, we let the end task model learn a linear combination of these representations.

To add ELMo to the supervised model, we first freeze the weights of the biLM and then concatenate the ELMo vector $$\text{ELMo}^{task}$$ with $$x_{k}$$ and pass the ELMo enhanced representation [$$x_{k}; \text{ELMo}^{task}$$] into task RNN.


## ULMFiT

The [paper](https://arxiv.org/pdf/1801.06146.pdf) by [Jermey Howard](http://twitter.com/jeremyphoward/) and [Sebestain Ruder](http://twitter.com/seb_ruder/) proposes a transfer learning method in NLP similar to the one which we saw in our previous blog on [Transfer Learning](https://dudeperf3ct.github.io/transfer/learning/catsvsdogs/2018/11/20/Power-of-Transfer-Learning/) on images. *So cool!*

There was a simple transfer learning technique involved in fine-tuning pretrained word embeddings and also approaches of ELMo and CoVe that concatenate embeddings derived from other tasks with the input at different layers but that only targets model's first layer barely scratching the surface of model for finetuning as seen in Computer Vision. These approaches mainly transfer word-level information instead of transferring high-level semantics. The authors argued that not the idea of LM fine-tuning but our lack of knowledge of how to train them effectively has been hindering wider adoption. 

### How it Works?

Universal Language Model Fine-tuning (ULMFiT) is the model that addresses the issues mentioned above and enables robust inductive transfer learning for any NLP task.

<p align="center">
<img src='/images/tl_nlp/ulmfit.png' width="70%"/>
</p>

ULMFiT consists of three stages: 

1. **General-domain LM pretraining** : Typical routine for creating pretraining vision models is to train on very large corpus of data (ImageNet size) and then use that freezed model as starting base model for finetuning. Similarly, Wikitext-103 consisting of 28,595 preprocessed Wikipedia articles and 103 million words is used to pretrain a language model. A language model as we discussed in ELMo section learns to predict next word in sentence. This prediction task makes language model more efficient in understanding grammar, semantics and other elements of corpus it is trained on. The base pretrained language model model is [AWD-LSTM](http://nlpprogress.com/english/language_modeling.html) described in another [paper](https://arxiv.org/pdf/1708.02182.pdf) by group at Salesforce, Merity et al. This is only step that needs to be performed once (to obtain pretrained model on large corpus) and is expensive step.

2. **Target task LM fine-tuning** :  As we know that data on target task and general-domain data used for pretraining can be different (come from a different distribution). This step will finetune LM data on target data. As noted above in lack of knowledge on how to train effectively is holding this process of transfer learning in nlp. To stabilize finetuning process, the authors propose two methods : a) Discriminative fine-tuning  and b) Slanted Triangular learning rates.

a) **Discriminative fine-tuning** : We have seen in [visualizing layer](https://dudeperf3ct.github.io/visualize/cnn/catsvsdogs/2018/12/02/Power-of-Visualizing-Convolution-Neural-Networks/) how different layers capture different types of information and also in biLM in ELMo. In Discriminative fine-tuning, each layer is updated using different learning rate {$$\eta^{1}, ..\eta^{L}$$}  for L layers in model where $$\eta^{l}$$ is learning rate of l-th layer. In practise, choosing the learning rate $$eta^{L}$$ of the last layer by fine-tuning only the last layer and using $$eta^{l-1}$$ = $$eta^{l}$$/2.6 as the learning rate for lower layers is found to work well.


b) **Slanted Triangular learning rates**: Using the same learning rate (LR) or an annealed learning rate throughout training is not the best way to achieve this behaviour. Instead, authors propose slanted triangular learning rates(STLR), which first linearly increases the learning rate and then linearly decays it according to the following update schedule.

<p align="center">
<img src='/images/tl_nlp/slr.png' width="60%"/>
</p>

where T is number of iteration (number of epochs x number of updates per epoch) and *cut_frac* is the fraction of iterations we increase the LR *cut* is the iteration when we switch fromincreasing to decreasing the LR, p is the fraction of the number of iterations we have increased or will decrease the LR respectively, ratio specifies how much smaller the lowest LR is from the maximum LR $$\eta_{max}$$ and $$\eta_{t}$$ is learning rate at iteration t. In practise, ratio = 32, *cut_frac* = 0.1 and $$\eta_{max}$$ = 0.01 is used.

3. **Target task classifier fine-tuning** : 

For finetuning classifier, pretrained language model is augmented with two additional linear blocks, a) concat pooling and b) gradual unfreezing.

a) **Concat pooling**:  The authors state that as input document can consist of hundreds of words, information may get lost if we only consider the last hidden state of the model. For this reason, we concatenate the hidden state at the last time step $$h_{T}$$ of the document with both the max-pooled and the mean-pooled representation of the hidden states over as many time steps as fit in GPU memory. If $$\mathcal{H}$$ = [h_{1},...,h_{T}]$$, then $$h_{c} = [h_{T}, \text{maxpool}(\mathcal{H}), \text{meanpool}(\mathcal{H})]$$.


b) **Gradual Unfreezing**: Rather than fine-tuning all layers at once, which may result in catastrophic forgetting, authors propose gradual unfreezing starting from last layer as it contains least amount of information. The steps involved are: We first unfreeze the last layer and fine-tune all unfrozen layers for one epoch. We then unfreeze the next lower frozen layer and repeat, until we fine-tune all layers until convergence at the last iteration. 

### TL;DR

- Wooh, CV transfer learning style training. Create a pretrained language model by training on large corpus like Wikitext-103, etc.
- Finetune LM data on target data and to stabalize this finetuning two methods like Discriminative finetuning and Slanted learning rates are used.
- To make target task classifier, additional linear model is added to language model architecture such as concat pooling is added and gradual unfreezing is used.

### Results

<p align="center">
<img src='/images/tl_nlp/jph_tweet.png' width="60%"/>
</p>

ULMFiT method significantly outperforms the SOTA on six text classification tasks, reducing the error by 18-24% on the majority of datasets. 

<p align="center">
<img src='/images/tl_nlp/ulmfit_result_1.png' width="70%"/>
</p>

<p align="center">
<img src='/images/tl_nlp/ulmfit_result_2.png' width="70%"/>
</p>


### What this means?

Ooooh, this is very exiciting. SoTA on everything! Take my money already.

ULMFiT shows one of the best approaches to tackling difficult problem through concatinating different methods into one. Transfer Learning has certainly change Computer Vision field and this method surely opens the door for similar breakthroughs in NLP field.

---

Before procedding to GPT and BERT, it is necessary to understand Transformer architecture properly introduced in paper "[Attention is All You Need](https://arxiv.org/pdf/1706.03762)". Here are recommended very cool resources other than paper to get you started

**Note**: [Dissceting Bert](https://medium.com/dissecting-bert) on medium dissects BERT and Transformer, for in-depth understanding BERT Encoder look here [part-1](https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3) and [part-2](https://medium.com/dissecting-bert/dissecting-bert-part2-335ff2ed9c73), Decoder of Transformer architecture look [here](https://medium.com/dissecting-bert/dissecting-bert-appendix-the-decoder-3b86f66b0e5f). 

**Note**: [keitakurita](http://mlexplained.com/author/admin/)  does a great job in dissecting the paper on the [blog](http://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/).

**Note**: Harvard NLP group has excellent [blog](http://nlp.seas.harvard.edu/2018/04/03/attention.html) detailing the paper "Attention is All You Need" which describes the Transformer architecture used by GPT and BERT with implementation details.

---

## GPT

The group at OpenAI proposed a new method [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) large gains on various nlp tasks can be realized by generative pretraining of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task. The main goal of paper was to learn a universal representation that transfers with little adaptation to a wide range of tasks.


### How it works?

GPT is short for Generative Pretraining Transformers. GPT uses a combination of unsupervised pretraining and supervised fine-tuning. Unlike ULMFit, the authors thought lets turn to Transformer architectures instead of Recurrent architectures for creating language models.

GPT training procedure consists of two steps:

- **Unsupervised pretraining**

GPT similar to ELMo uses a standard language model, where instead of using biLM model i.e. both forward and backward direction, GPT uses only forward direction and the model architecture is multi-layer Transformer decoder adapted from this [paper](https://arxiv.org/pdf/1801.10198.pdf) for language model. This model applies multiple transformer blocks over the embeddings of input sequences. Each block contains a masked multi-headed self-attention layer and a pointwise feed-forward layer. The final output produces a distribution over target tokens after softmax normalization.

$$
\begin{aligned}
h_{0} & = UW_{e} + W_{p} \\
h_{l} &  = transformer\_block(h_{l-1}) \\
P(u) &  = softmax(h_{n}W_{e}^{T}) \\
\end{aligned}
$$

where $$W_{e}$$ is token embedding matrix, $$W_{p}$$ is position embedding matrix, n is number of layers and U = ($$U_{-k}... U_{-1}$$) is the context vector of tokens.

The objective to maximize as seen in ELMo will be the only forward direction of biLM.

$$
\begin{aligned}
\mathcal{L}_{LM} = \sum_{k=1}^{N}(log (p(t_{k} \mid t_{1}, t_{2}, ..., t_{k-1}))
\end{aligned}
$$

[Byte Pair Encoding](https://arxiv.org/pdf/1508.07909) (BPE) is used to encode the input sequences. Motivated by the intuition that rare and unknown words can often be decomposed into multiple subwords, BPE finds the best word segmentation by iteratively and greedily merging frequent pairs of characters.

<p align="center">
<img src='/images/tl_nlp/gpt_transformer.png' width="70%"/>
</p>

- **Semi-supervised learning for NLP**

After training with objective $$\mathcal{L}_{LM}$$, the inputs where each instance consists of a sequence of input tokens, $$x^{1}, x^{2} ..., x^{m}$$ along with label y are passed through our pretrained model to obtain the final transformer block’s activation $$h_{l}^{m}$$ which is then fed into an added linear output layer with parameters $$W_{y}$$ to predict y:

$$
\begin{aligned}
(P(y \mid x^{1}, x^{2}, ..., x^{m}) & = softmax(h_{l}^{m}W_{y}) \\
\mathcal{L}_{C} &  = \sum_{(x,y)}^{}(log (P(y \mid x^{1}, x^{2}, ..., x^{m})) \\
\mathcal{L}_{total} &  = \mathcal{L}_{C} + \lambda * \mathcal{L}_{LM}
\end{aligned}
$$

GPT gets rid of any task-specific customization or any hyperparameter tuning when applying across various tasks. If the task input contains multiple sentences, a special delimiter token ($) is added between each pair of sentences. The embedding for this delimiter token is a new parameter we need to learn, but it should be pretty minimal. All transformations include adding randomly initialized start and end tokens (〈s〉,〈e〉).


<p align="center">
<img src='/images/tl_nlp/gpt_rid.png' width="70%"/>
</p>


### TL;DR

- GPT makes use of unlabelled data to train a language model using a multi-layer Transformer decoder architecture. 
- Langauge model pretrained above can be applied across various tasks directly instead of training different langauge models across different tasks.


### Results

That's a lot of results. GPT significantly improves upon the SOTA in 9 out of the 12 tasks.

<p align="center">
<img src='/images/tl_nlp/gpt_result.png' width="60%"/>
</p>


### What this means?

By pretraining on a diverse corpus with long stretches of contiguous text our model acquires significant world knowledge and ability to process long-range dependencies which are then successfully transferred to solving discriminative tasks such as question answering, semantic similarity assessment, entailment determination, and text classification, improving the state of the art. The advantage of this approach is few parameters need to be learned from scratch.

One limitation of GPT is its unidirectional nature — the model is only trained to predict the future left-to-right context.


## BERT

Yo myself, BERT. I will improve the shortcomings of GPT.

<p align="center">
<img src='/images/tl_nlp/bert.jpg' width="40%"/>
</p>


### How this works?

BERT stands for Bidirectional Encoder Representations for Transformers. [BERT](https://arxiv.org/pdf/1810.04805.pdf) is designed by group at Google AI Language to pretrain deep bidirectional representations by jointly conditioning on both left and right context in all layers. With adding different output layers to pretrained BERT, this model can be used for various nlp tasks.  

We have seen two strategies for applying pretrained language representations to downstream tasks : feature-based and fine-tuning. ELMo is example of feature-based where various task-specific architectures are used as additional features and GPT is example of fine-tuning which has minimal task-specific parameters is trained on the downstream tasks by simply fine-tuning the pre-trained  parameters.

<p align="center">
<img src='/images/tl_nlp/diff.png' width="70%"/>
</p>

Here are the differences in pretraining model architectures. BERT uses bidirectional Transformer. OpenAI GPT uses a left-to-right Transformer. ELMo uses the concatenation of independently trained left-to-right and right-to-left LSTM to generate features for downstream tasks.  BERT Transformer uses  bidirectional  self-attention, while the GPT Transformer uses constrained self-attention where every token can only attend to context to its left. In the literature the bidirectional Transformer is often referred to as a “Transformer encoder” while the left-context-only version is referred to as a “Transformer decoder” since it can be used for text generation.


<p align="center">
<img src='/images/tl_nlp/bert_pretraining.png' width="30%"/>
</p>

The authors argue that GPT used left-to-right architecture on standard langauge model is limiting in choice and a deep  bidirectional model is strictly more powerful than either a left-to-right model (GPT) or the shallow concatenation of a left-to-right and right-to-left model (ELMo). The authors propose a new language model with new objective: "masked language model"(MLM) and "next sentence prediction".

Input to BERT is composed of multiple parts: (i) Token Embeddings Use of [WordPiece](https://arxiv.org/pdf/1609.08144.pdf) embeddings with a 30,000 token vocabulary and denote split word pieces with ## (ii) Position Embeddings: learned positional embeddings with supported sequence lengths upto 512 tokens (iii) The first token of every  sequence is always the special classification embedding([CLS]) (iv) Segment Embeddings: Sentence pairs are packed together into a single sequence.  We differentiate the sentences in two ways. First, we separate them  with a special token ([SEP]). Second, we add a learned sentence A embedding to every token of the first sentence and a sentence B embedding to every token of the second sentence, and for single-sentence inputs we only use the sentence A embeddings.
 
BERT's input representation is constructed by summing the corresponding token, segment and position embeddings.


<p align="center">
<img src='/images/tl_nlp/bert_input.png' width="70%"/>
</p>

Similar to GPT, BERT training takes place in two steps:

- **Pretraining tasks**:  Unlike GPT, BERT's model architecture is multi-layer bidirectional Transformer encoder. To encourage the bidirectional prediction and sentence-level understanding, BERT is trained with two auxiliary tasks (masking random words and next sentence prediction) instead of the basic language task (that is, to predict the next token given context).

a) **Task #1: Masked LM**: Here we mask some percentage of the input tokens at random, and then predict only those masked tokens. Consider for example sentence: *my dog is hairy*. Here, it chooses *hairy*. It randomly masks 15% of tokens in a sequence and rather than always replacing the chosen words with [MASK], the data generator will do the following: (i) Replace the word with [MASK] token 80% of time i.e. *my dog is hairy → my dog is [MASK]*, (ii) Replace the word with a random word 10% of time i.e. *my dog is hairy → my dog is apple*, (iii) Keep the word untouched 10% of time i.e. *my dog is hairy → my dog is hairy*. The purpose of this is to bias the representation towards  the actual observed word. The Transformer encoder does not know which words it  will be asked to predict or which have been replaced by random words. This forces LM to keep a distributional contextual representation of every input token.

b) **Task #2: Next Sentence Prediction**: In order to train a model that understands sentence relationships which can be useful for downstream tasks such as Question Answering (QA) and Natural Language Inference (NLI), we train a model to capture this relationship in language model. When choosing sentence A and B for each pretraining example, 50% of time B is actual next sentence that follows A, and 50% of time it is a random sentence from corpus. The final pretrained model achieves 97%-98% accuracy at this task.

- **Finetuning Procedure**: For classification task, we take the final hidden state (i.e. the output of Transformer) for the first token in input which is special token [CLS], $$h_{L}^{CLS}$$, and multiply it with weight matrix of classification layer $$W_{CLS}$$ which is the only added parameter during fine-tuning. Then the label probabilities is applying standard softmax which is $$P = softmax(h_{L}^{CLS} W_{CLS}^{T})$$. For other downstream tasks, following figure explains some task-specific modification to be made.

<p align="center">
<img src='/images/tl_nlp/bert_cls_1.png' width="70%"/>
</p>

<p align="center">
<img src='/images/tl_nlp/bert_cls_2.png' width="70%"/>
</p>

Understanding and choosing correct hyperparameters(*there are too many*) can make or break BERT. So, we need to choose wisely. Paper outlines some experiements which I would encourage the curious readers to have a look.

[Paper](https://arxiv.org/pdf/1810.04805.pdf) also outlines differences between BERT and GPT.

- GPT is trained on the BooksCorpus (800M words); BERT is trained on the BooksCorpus (800M words) and Wikipedia (2,500M words).
- GPT uses a sentence separator ([SEP]) and classifier token ([CLS]) which are only introduced at fine-tuning time; BERT learns [SEP], [CLS] and sentence A/B embeddings during pre-training.
- GPT was trained for 1M steps with a batchsize of 32,000 words; BERT was trained for 1M steps with a batch size of 128,000 words.
- GPT used the same learning rate of 5e-5 for all fine-tuning experiments; BERT chooses a task-specific fine-tuning learning rate which performs the best on the development set.


### TL;DR

- Use large corpus of unlabeled data to learn a language model(which captures semantics, etc of language) by training on two tasks: Masked Language Model and Next Sentence Prediction using a multi-layer bidirectional Transformer Encoder architecture.
- Finetuning pretrained language model for specific downstream tasks, task-specific modifications are done.


### Results

Hold on, here comes the result. BERT outperforms previous SOTA in 11 tasks. Yay!! Go, BERT.

<p align="center">
<img src='/images/tl_nlp/bert_results.png' width="70%"/>
</p>

### What this means?

This means BERT is super cool, that's it! We can use pretrained [BERT models](https://github.com/google-research/bert#pre-trained-models) to finetune for specific tasks.


## GPT-2

Look who shows up at showdown in between GPT and BERT, GPT's big brother GPT-2. OpenAI team introduces next version of GPT in the [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), GPT-2. 

### How it Works?

GPT-2 is a large transformer-based language model with 1.5 billion parameters (10x more than GPT), trained on a dataset of 8 million web pages. GPT-2 is trained with a simple objective: predict the next word, given all of the previous words within some text.

The authors state that the [paper]() from Google AI which performed Multi-task Learning on .. tasks required supervision but language modeling, in principle is able to learn such task without the need for explicit supervision. Authors perform preliminary experiments to confirm that sufficiently large language models are able to perform multitask learning in toy-ish setup but learning is much slower than in explicitly supervised approaches. 

The internet contains a vast amount of information that is passively available without the need for interactive communication like in dialog or QA tasks. Authors speculate that a language model with sufficient capacity will begin to learn to infer and  perform the tasks demonstrated in natural language sequences in order to better predict them, regardless of their method of procurement. If a language model is able to do this it will be, in effect, performing unsupervised multitask learning. Authors propose using Zero-shot Transfer by pretraining a language model on various tasks and conditioning tasks along with input to get task-specific output, p(*output*|*input*,*task*) instead of finetuning for seperate tasks where for each task the conditional probability is p(*output*|*input*). 

- **Zero-shot Transfer** : GPT-2 learns it's language model on diverse dataset in order to collect natural language demonstrations of tasks in as varied of domains and contexts as possible. While preprocessing LM, authors state that current byte-level LMs are not competitive with word-level LMs on large scale datasets. They modify BPE (Byte Pair encoding) to combine benefits word-level LM with the generality of byte-level approaches. 

- **Byte Pair Encoding** : Byte Pair Encoding (BPE) is a practical middle ground between character and word level language modeling which effectively interpolates between word level inputs for frequent symbol sequences and character level inputs for infrequent symbol sequences. BPE implementations often operate on Unicode code points and not byte sequences. Each byte can represent 256 different values in 8 bits, while UTF-8 can use up to 4 bytes for one character, supporting up to 231 characters in total. Therefore, with byte sequence representation we only need a vocabulary of size 256 and do not need to worry about pre-processing, tokenization, etc. BPE merges frequently co-occurred byte pairs in a greedy manner. To prevent it from generating multiple versions of common words (i.e. dog., dog! and dog? for the word dog), GPT-2 prevents BPE from merging characters across categories (thus dog would not be merged with punctuations like ., ! and ?). This tricks improves the compression efficiency while adding only minimal fragmentation of words across multiple vocab tokens.

GPT-2 follows similar Transformer architecture used in GPT. The model details is largely similar to GPT model with a few modifications: Layer normalization was moved to  the input of each sub-block, similar to a pre-activation residual network and an additional layer normalization was added after the final self-attention block, a modified initialization was constructed as a function of the model depth, scaling the weights of residual layers at initialization by a factor of $$1/ \sqrt{N}$$ where N is the number of residual layers. The vocabulary is expanded to 50,257 and also increase the context size from 512 to 1024 tokens and a larger batch size of 512 is used.

**Downstream Tasks**

- **Text Generation** : Text generation is standard given pretrained LM. Here is one example of text generation

<p align="center">
<img src='/images/tl_nlp/gpt_2_text.png' width="70%"/>
</p>

*So real but not real! or is it?*

- **Summarization** : Adding TL;DR after articles produces summary.

<p align="center">
<img src='/images/tl_nlp/gpt_2_summary.png' width="70%"/>
</p>

- **Machine Translation** : Using conditional probability of target language, translation is obtained. For e.g. for translating English to Chinese P(? | I like green apples. = 我喜欢绿苹果。 A cat meows at him. = 一只猫对他喵。It is raining cats and dogs. =") will give the translation of "It is raining cats and dogs." in Chinese.


<p align="center">
<img src='/images/tl_nlp/gpt_2_translate.png' width="70%"/>
</p>


- **Question Answering** : Similar to translation, pairs of question and answer and context can be conditioned to give the answer for required question.

<p align="center">
<img src='/images/tl_nlp/gpt_2_qa.png' width="70%"/>
</p>



### TL;DR

- Large and diverse amount data is enough to capture language semantics related to different tasks instead of training a language model for seperate tasks.
- Pretrained lanaguage model does excellent job on various tasks such as question answering, machine translation, summarization and especially text generation without having to train explicitly for each particular tasks. No task-specific finetuning required.
- GPT-2 achieves mind blowing results just through pretrained language model.


### Results

I bet results would be SOTA and they are, on 7 tasks out of 8.

<p align="center">
<img src='/images/tl_nlp/gpt_2_results.png' width="80%"/>
</p>


### What this means?

Just training LM (*no task-specific finetuning*) that is all it took. Results are mind (*into tiny pieces*) blowing.





<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology


---

# Further Reading

Must Read! Awesome Lil'Log [Generalized Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html)

Must Read! [Dissecting BERT Part 1: The Encoder](https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3), [Understanding BERT Part 2: BERT Specifics](https://medium.com/dissecting-bert/dissecting-bert-part2-335ff2ed9c73) and [Dissecting BERT Appendix: The Decoder](https://medium.com/dissecting-bert/dissecting-bert-appendix-the-decoder-3b86f66b0e5f)

Must Read! Jay Alammar [Illustrated bert](http://jalammar.github.io/illustrated-bert/), [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) and [Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

Must Read! mlexplained.com Paper Dissected [BERT](http://mlexplained.com/2019/01/07/paper-dissected-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-explained/), [ELMo](http://mlexplained.com/2018/06/15/paper-dissected-deep-contextualized-word-representations-explained/) and [Attention Is All You Need](http://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/)

Must Read! Havard NLP [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

[Salesforce Research Blog: CoVe](https://blog.einstein.ai/learned-in-translation-contextualized-word-vectors/)

[CoVe](https://arxiv.org/pdf/1708.00107.pdf)

[ELMo](https://arxiv.org/pdf/1802.05365.pdf)

[ELMo blog by Masato Hagiwara](http://www.realworldnlpbook.com/blog/improving-sentiment-analyzer-using-elmo.html)

[ULMFiT](https://arxiv.org/pdf/1801.06146.pdf)

[Fastai blog on ULMFiT](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)

[Attention is All You Need](https://arxiv.org/pdf/1511.01432)

[Semi-supervised Sequence Learning](https://arxiv.org/pdf/1511.01432)

[Byte Pair Encoding](https://arxiv.org/pdf/1508.07909)

[GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

[OpenAI GPT Blog](https://blog.openai.com/language-unsupervised/)

[BERT](https://arxiv.org/pdf/1810.04805.pdf)

[Google AI blog BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)

[GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

[OpenAI GPT-2 Blog](https://openai.com/blog/better-language-models/)

---

# Footnotes and Credits


[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)

[Meme src](https://twitter.com/gregd_nlp/status/1096244878600818693)

[ELMO](https://twitter.com/elmo)

[ELMo biLM](http://www.realworldnlpbook.com/blog/improving-sentiment-analyzer-using-elmo.html)

[ELMo biLSTM](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html)

---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

