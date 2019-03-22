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

> All the codes implemented in Jupyter notebook in [Keras](https://github.com/dudeperf3ct/DL_notebooks/blob/master/lstm_and_gru/lstm_and_gru_keras.ipynb), [PyTorch](https://github.com/dudeperf3ct/DL_notebooks/blob/master/lstm_and_gru/lstm_and_gru_pytorch.ipynb), [Flair](https://github.com/dudeperf3ct/DL_notebooks/blob/master/lstm_and_gru/lstm_and_gru_flair.ipynb) and [fastai](https://github.com/dudeperf3ct/DL_notebooks/blob/master/lstm_and_gru/lstm_and_gru_fastai.ipynb).

> *All codes can be run on Google Colab (link provided in notebook).*

Hey yo, but how?

Well sit tight and buckle up. I will go through everything in-detail.


Feel free to jump anywhere,

- [](#nlp-tasks)
- [Further Reading](#further-reading)
- [Footnotes and Credits](#footnotes-and-credits)

## NLP Tasks


### Sentiment analysis


### POS


### NER


### Textual entailment


### Coreference resolution



### Question Answering



# Transfer Learning in NLP

<span class='blue'> A long time ago in a galaxy far, far away.... </span>

<p align="center">
<img src='/images/transfer_learning_files/master_student.gif' /> 
</p>


<span class='red'>I-know-everything:</span> 


<span class='green'>I-know-nothing:</span> 


<span class='red'>I-know-everything:</span> 

The [embedding models](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#embeddings) which we disscused earlier like [word2vec](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#word2vec), [GLoVe](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#glove) and [fastText](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#fasttext) are fantastic in capturing meaning of individual words and their relationships by leveraging large datasets. These model generate word vectors of n-dimension which is used by neural network as starting point of training. The word vectors can be initialized to lists of random numbers before a model is trained for a specific task, or initialized with word vectors obtained from above embedding models.

A word is assigned the same vector representation no matter where it appears and how it's used, because word embeddings rely on just a look-up table. In other word, they ignore polysemy — a concept that words can have multiple meanings. 



## CoVe

In NLP tasks, context matters. That is, understanding context is very essential to all NLP tasks as words rarely appear in isolation and also helps in general sense of language understanding tasks. One such example is in Question Answering where understanding of how words in question shift the importance of words in document or in Summarization where model needs to understand which words capture the context clearly to summarize succinctly. The ability to share a common representation of words in the context of sentences that include them could further improve transfer learning in NLP. This is where CoVe comes into play which transfers information from large amounts of unlabeled training data in the form of word vectors using encoder to contextualize word vector has shown to improve performance over random word vector initialization on a variety of downstream tasks e.g. POS, NER and QA. 

### How it Works?

CoVe is Contextual Word Vectors, type of word embedding learned by encoder in an attentional seq-seq machine translational model.
The team at Salesforce explained CoVe in best way on their [research blog](https://blog.einstein.ai/learned-in-translation-contextualized-word-vectors/), also outlined in [their paper](https://arxiv.org/pdf/1708.00107.pdf). We will look at a special case example of Machine Translation from English (source language) to German (target language).

-cove.png



- **Encoder**

A neural network [BiLSTM](https://dudeperf3ct.github.io/lstm/gru/nlp/2019/01/28/Force-of-LSTM-and-GRU/#bidirectional-lstm-network) takes word vectors as input and outputs a new vector called hidden vector. This process is often referred to as encoding the sequence, and the neural network that does the encoding is referred to as an encoder. BiLSTM (forward and backward LSTM) is used to incorporate information from words that appear later in the sequence.

-enocder_cove

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

-decoder_cove

Tbe decoder takes in input randomly intialized embedding for target words $$w^{z}$$ = [$$w^{z}_{1}, w^{z}_{2}, ..., w^{z}_{n}$$] , context-adjusted state generated by attention mechanism and previous hidden state of LSTM. Similar to encoder, decoder uses two layer LSTM (unidirectional) to create decoder state from input word vectors.


$$
\begin{aligned}
\textbf{Decoder hidden state} : s_{t} = LSTM([w^{z}_{t-1}; \tilde{h}_{t-1}], s_{t-1})
\end{aligned}
$$

- **Attention**

Attention mechanism is one the interesting mechanism in NLP and we will look into more depth in next post. The attention mechanism looks back at hidden vectors in order to decide which part of English sentence to translate next. It uses the state vector to determine how important each hidden vector is, and then it produces a new vector, which we will call the context-adjusted state, to record its observation.

-attention_cove

Attention mechanism uses hidden vectors generated by encoder and decoder state by decoder to produce context-adjusted state. It sort of plays a role of deciding which context words play important role in translation and focusing on them rather than whole sentence.

$$
\begin{aligned}
\textbf{Attention Weights} : \alpha_{t} & = softmax(H(W_{1}s_{t} + b_{1}))\\ 
\textbf{Context-adjusted Weights} : \tilde{h}_{t} & = tanh(W_{2}[H^{T} \alpha_{t}; s_{t}] + b_{2})\\
\end{aligned}
$$

Here H is is a stack of hidden states {h} along the time dimension.

- **Generation**

The generator then looks at the context-adjusted state to determine which German word to output, and the context-adjusted state is passed back to the decoder so that it has an accurate sense of what it has already translated. The decoder repeats this process until it is done translating. This is a standard attentional encoder-decoder architecture for learning sequence to sequence tasks like machine translation.

-generator_cove

The generator uses context-adjusted state from attention mechanism to produce output German word. Attention mechanism takes in input all hidden vectors and first decoder state to produce context-adjusted state which will be used as input to decoder to produce another decoder state. This decoder state along with all hidden vectors will again be used as input to attention mechanism to generate another context-adjusted state which will be input to third decoder and so on.

$$
\begin{aligned}
\textbf{Generator Output} : p(y_{t} \mid H,y_{1},y_{2}, …,y_{t-1}) & = softmax(W_{out}\tilde{h}_{t} + b_{out}))
\end{aligned}
$$

Here $$p(y_{t} \mid H,y_{1},y_{2}, …,y_{t-1})$$ is a probability distribution over output words.

### Results

Hmm, that seems simple process but what about results? Was is it SoTA breaker?

-cove_results.png

Here CoVe+GLove means that we take the GloVe sequence, run it through a pretrained MT-LSTM to get CoVe sequence, and we append each vector in the CoVe sequence with the corresponding vector in the GloVe sequence.

-cove_results_1.png


SoTA in 3 out of 7 tasks, well that's  a good start with using CoVe pretrained vectors.

-cove_results_overall.png

### What this means?

Replacing the good ol' GloVe, Word2vec and fastText with CoVe seems to do a good job at the tasks where context matters. Training a custom pretrained CoVe model is also simple. Just take any unlabelled data corresponding to task at hand (e.g. Amazon Review for SST or IMDB 50,000 unlabelled reviews for IMDb sentiment analysis task) pass it through encoder (MT-LSTM) to generate CoVe word vector in supervised fashion and we can use that CoVe pretrained vector along with GloVe vector as initial embedding model and use that train for specific task like sentiment analysis, Question Answering, Machine Translation, etc. The more data we use to train the MT-LSTM, the more pronounced the improvement, which seems to be complementary to improvements that come from using other forms of pretrained vector representations.

-general_cove.png

Here there is disadavantage of using only avaliable data for generating pretrained CoVe embedding using supervised training of encoder-decoder architecture. (*no large unsupervisied dataset which are everywhere, supervised learning requires labels too*)

## ELMo

Hi, my name is ELMo and I will overcome the limitation of CoVe by generating contextual embeddings in an unsupervised fashion.

-elmo.png

ELMo stands for Embeddings from Language Models. ELMo is a word representation technique proposed by [AllenNLP](https://arxiv.org/pdf/1802.05365.pdf)

### How it Works?

ELMo word representations are function of entire input sentence and are computed on top of two biLM with character convolutions.

- **Bidirectional Language Model**

-elmo_bilm.png

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
\mathcal{L} = \sum_{k=1}^{N}(log (p(t_{k} \mid t_{1}, t_{2}, ..., t_{k-1}); \Theta_{x},  \overset{\rightarrow}\Theta_\text{LSTM}, \Theta_{s}) \\ + log (p(t_{k} \mid t_{k+1}, t_{k+2}, ..., t_{N}); \Theta_{x},  \overset{\leftarrow}\Theta_\text{LSTM}, \Theta_{s}))
\end{aligned}
$$

Here $$\Theta_{x}$$ and $$\Theta_{s}$$ are embedding layers and softmax layers. Overall, this formulation is similar to the approach of CoVe, with the exception that we share some weights between directions instead of using completely independent parameters. The internal states of forward pass at a certain word reflect the word itself and what has happened before that word, whereas similar can be concluded for backward pass where word itself and what has happened after that word gets reflected. These two passes are concatenated to get intermediate word vector of that word. Therefore, this intermediate word vector at that word is still the representation of what the word means, but it "knows" what is happening (i.e. captures the essence or context) in the rest of the sentence and how the word is used.

-elmo_bilstm.png

ELMo uses two layer biLM where each biLM layer consists of one forward pass and one backward pass that scans the sentence in both directions. ELMo is a task specific combination of the intermediate layer representations in the biLM. For each token $$t_{k}$$, a L-layer biLM computes a set of L+1 representations:

$$
\begin{aligned}
\mathcal{R}_{k} & = \{x_{k}^{LM}, \overset{\rightarrow}h_{k,j}^{LM} ,\overset{\leftarrow}h_{k,j}^{LM} \mid j=1,2,...,L\} \\
& = \{h_{k, j}^{LM} \mid j = 0,1,2,...,L\}
\end{aligned}
$$

where $$h_{k, 0}^{LM}$$ is embedding layer $$h_{k, j}^{LM} = [\overset{\rightarrow}h_{k,j}^{LM} ; \overset{\leftarrow}h_{k,j}^{LM}]$$, for each biLSTM layer.

For inclusion in a downstream model, ELMO collapses all layer in R into a single vector,

$$
\begin{aligned}
\text{ELMo}_{k}^{task} & = E[\mathcal{R}_{k}; \Theta^{task}]\\
& = \gamma^{task} \sum_{j=0}^{L}s_{j}^{task}h_{k, j}^{LM}
\end{aligned}
$$

where $$s^{task}$$ are softmax-normalized weights and the scalar parameter $$\gamma^{task}$$ allows the task model to scale the entire ELMo vector.


## ULMFiT




## GPT



## BERT




## GPT-2





<span class='orange'>Happy Learning!</span>

---

### Note: Caveats on terminology


---

# Further Reading

[Salesforce Research Blog: CoVe](https://blog.einstein.ai/learned-in-translation-contextualized-word-vectors/)

[CoVe](https://arxiv.org/pdf/1708.00107.pdf)

[ELMo](https://arxiv.org/pdf/1802.05365.pdf)

[ULMFiT]()

[GPT]()

[BERT]()

[GPT-2]()

---

# Footnotes and Credits


[Star Wars gif](https://www.behance.net/gallery/30412489/Star-Wars-Luke-Yoda-R2D2-in-Dagobah-Animated-Gif)

[Meme src](https://twitter.com/gregd_nlp/status/1096244878600818693)

[ELMO](https://twitter.com/elmo)


---

**NOTE**

Questions, comments, other feedback? E-mail [the author](mailto:imdudeperf3ct@gmail.com)


---

