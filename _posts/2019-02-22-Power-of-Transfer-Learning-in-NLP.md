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
- [CoVe](#cove)
- [ELMo](#elmo)
- [ULMFiT](#ulmfit)
- [GPT](#gpt)
- [BERT](#bert)
- [GPT-2](#gpt-2)
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

### TL;DR



### Results

Hmm, that seems simple process but what about results? Was is it SOTA breaker?

-cove_results.png

Here CoVe+GLove means that we take the GloVe sequence, run it through a pretrained MT-LSTM to get CoVe sequence, and we append each vector in the CoVe sequence with the corresponding vector in the GloVe sequence.

-cove_results_1.png


SOTA in 3 out of 7 tasks, well that's  a good start with using CoVe pretrained vectors.

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

A language model is an NLP model which learns to predict the next word in a sentence. For instance, if your mobile phone keyboard guesses what word you are going to want to type next, then it’s using a language model. The reason this is important is because for a language model to be really good at guessing what you’ll say next, it needs a lot of world knowledge (e.g. “I ate a hot” → “dog”, “It is very hot” → “weather”), and a deep understanding of grammar, semantics, and other elements of natural language.

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
\mathcal{L}_{LM} = \sum_{k=1}^{N}(log (p(t_{k} \mid t_{1}, t_{2}, ..., t_{k-1}); \Theta_{x},  \overset{\rightarrow}\Theta_\text{LSTM}, \Theta_{s}) \\ + log (p(t_{k} \mid t_{k+1}, t_{k+2}, ..., t_{N}); \Theta_{x},  \overset{\leftarrow}\Theta_\text{LSTM}, \Theta_{s}))
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


### Results

Well, ELMo certainly outperforms CoVe and emerges as new SOTA at all the 6 tasks with relative error reductions ranging from 6 - 20%. 

-elmo_results.png


### What this means?

Here is one the results from context embedding of biLM.

-bilm_example.png

Notice how biLM s able to disambiguate both the part of speech and word sense in the source sentence of word "play" than glove counterpart which has fixed neighbours no matter the context.

ELMo improves task performance over word vectors as the biLM’s contextual representations encodes information generally useful for NLP tasks that is not captured in word vectors.

Once pretrained, the biLM can compute representations for any task. In some cases, fine tuning the biLM on domain specific data leads to significant drops in perplexity and an increase in down-stream task performance. 

Given a pretrained LM and a supervised architecture for a target NLP task, it is a simple process to use the biLM to improve the task model. We simply run the biLM and record all of the layer representations for each word. Then, we let the end task model learn a linear combination of these representations.

To add ELMo to the supervised model, we first freeze the weights of the biLM and then concatenate the ELMo vector $$\text{ELMo}^{task}$$ with $$x_{k}$$ and pass the ELMo enhanced representation [$$x_{k}; \text{ELMo}^{task}$$] into task RNN.


## ULMFiT

The [paper](https://arxiv.org/pdf/1801.06146.pdf) by [Jermey Howard]() and [Sebestain Ruder](http://twitter.com/seb_ruder/) proposes a transfer learning method in NLP similar to the one which we saw in our previous blog on [Transfer Learning](https://dudeperf3ct.github.io/transfer/learning/catsvsdogs/2018/11/20/Power-of-Transfer-Learning/) on images. *So cool!*

There was a simple transfer learning technique involved in fine-tuning pretrained word embeddings and also approaches of ELMo and CoVe that concatenate embeddings derived from other tasks with the input at different layers but that only targets model's first layer barely scratching the surface of model for finetuning as seen in Computer Vision. These approaches mainly transfer word-level information instead of transferring high-level semantics. The authors argued that not the idea of LM fine-tuning but our lack of knowledge of how to train them effectively has been hindering wider adoption. 


### How it Works?

Universal Language Model Fine-tuning (ULMFiT) is the model that addresses the issues mentioned above and enables robust inductive transfer learning for any NLP task.

-ulmfit.png

ULMFiT consists of three stages: 

1. **General-domain LM pretraining** : Typical routine for creating pretraining vision models is to train on very large corpus of data (ImageNet size) and then use that freezed model as starting base model for finetuning. Similarly, Wikitext-103 consisting of 28,595 preprocessed Wikipedia articles and 103 million words is used to pretrain a language model. A language model as we discussed in ELMo section learns to predict next word in sentence. This prediction task makes language model more efficient in understanding grammar, semantics and other elements of corpus it is trained on. The base pretrained language model model is [AWD-LSTM](http://nlpprogress.com/english/language_modeling.html) described in another [paper](https://arxiv.org/pdf/1708.02182.pdf) by group at Salesforce, Merity et al. This is only step that needs to be performed once (to obtain pretrained model on large corpus) and is expensive step.

2. **Target task LM fine-tuning** :  As we know that data on target task and general-domain data used for pretraining can be different (come from a different distribution). This step will finetune LM data on target data. As noted above in lack of knowledge on how to train effectively is holding this process of transfer learning in nlp. To stabilize finetuning process, the authors propose two methods : a) Discriminative fine-tuning  and b) Slanted Triangular learning rates.

a) **Discriminative fine-tuning** : We have seen in [visualizing layer](https://dudeperf3ct.github.io/visualize/cnn/catsvsdogs/2018/12/02/Power-of-Visualizing-Convolution-Neural-Networks/) how different layers capture different types of information and also in biLM in ELMo. In Discriminative fine-tuning, each layer is updated using different learning rate {$$\eta^{1}, ..\eta^{L}$$}  for L layers in model where $$\eta^{l}$$ is learning rate of l-th layer. In practise, choosing the learning rate $$eta^{L}$$ of the last layer by fine-tuning only the last layer and using $$eta^{l-1}$$ = $$eta^{l}$$/2.6 as the learning rate for lower layers is found to work well.


b) **Slanted Triangular learning rates**: Using the same learning rate (LR) or an annealed learning rate throughout training is not the best way to achieve this behaviour. Instead, authors propose slanted triangular learning rates(STLR), which first linearly increases the learning rate and then linearly decays it according to the following update schedule.

-slr.png

where T is number of iteration (number of epochs x number of updates per epoch) and *cut_frac* is the fraction of iterations we increase the LR *cut* is the iteration when we switch fromincreasing to decreasing the LR, p is the fraction of the number of iterations we have increased or will decrease the LR respectively, ratio specifies how much smaller the lowest LR is from the maximum LR $$\eta_{max}$$ and $$\eta_{t}$$ is learning rate at iteration t. In practise, ratio = 32, *cut_frac* = 0.1 and $$\eta_{max}$$ = 0.01 is used.

3. **Target task classifier fine-tuning** : 

For finetuning classifier, pretrained language model is augmented with two additional linear blocks, a) concat pooling and gradual b) unfreezing.

a) **Concat pooling**:  The authors state that as input document can consist of hundreds of words, information may get lost if we only consider the last hidden state of the model. For this reason, we concatenate the hidden state at the last time step $$h_{T}$$ of the document with both the max-pooled and the mean-pooled representation of the hidden states over as many time steps as fit in GPU memory. If $$\mathcal{H}$$ = [h_{1},...,h_{T}]$$, then $$h_{c} = [h_{T}, \text{maxpool}(\mathcal{H}), \text{meanpool}(\mathcal{H})]$$.


b) **Gradual Unfreezing**: Rather than fine-tuning all layers at once, which may result in catastrophic forgetting, authors propose gradual unfreezing starting from last layer as it contains least amount of information. The steps involved are: We first unfreeze the last layer and fine-tune all unfrozen layers for one epoch. We then unfreeze the next lower frozen layer and repeat, until we fine-tune all layers until convergence at the last iteration. 

### TL;DR




### Results

-jph_tweet.png

ULMFiT method significantly outperforms the SOTA on six text classification tasks, reducing the error by 18-24% on the majority of datasets. 

-ulmfit_result_1.png

-ulmfit_result_2.png


### What this means?

Ooooh, this is very exiciting. SoTA on everything! Take my money already.

ULMFiT shows one of the best approaches to tackling difficult problem through concatinating different methods into one. Transfer Learning has certainly change Computer Vision field and this method surely opens the door for similar breakthroughs in NLP field.


## GPT

The group at OpenAI proposed a new method [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) large gains on various nlp tasks can be realized by generative pretraining of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task. The main goal of paper was to learn a universal representation that transfers with little adaptation to a wide range of tasks.


### How it works?

GPT is short for Generative Pretraining Transformers. GPT uses a combination of unsupervised pretraining and supervised fine-tuning. 

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

where W_{e} is token embedding matrix, W_{p} is position embedding matrix, n is number of layers and U = ($$U_{-k}... U_{-1}$$) is the context vector of tokens.

The objective to maximize as seen in ELMo will be the only forward direction of biLM.

$$
\begin{aligned}
\mathcal{L}_{LM} = \sum_{k=1}^{N}(log (p(t_{k} \mid t_{1}, t_{2}, ..., t_{k-1}))
\end{aligned}
$$

[Byte Pair Encoding](https://arxiv.org/pdf/1508.07909) (BPE) is used to encode the input sequences. Motivated by the intuition that rare and unknown words can often be decomposed into multiple subwords, BPE finds the best word segmentation by iteratively and greedily merging frequent pairs of characters.

-gpt_transformer.png

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

-gpt_rid.png



### TL;DR

- 


### Results

That's a lot of results. GPT significantly improves upon the SOTA in 9 out of the 12 tasks.

-gpt_result.png




### What this means?

By pretraining on a diverse corpus with long stretches of contiguous text our model acquires significant world knowledge and ability to process long-range dependencies which are then successfully transferred to solving discriminative tasks such as question answering, semantic similarity assessment, entailment determination, and text classification, improving the state of the art.

One limitation of GPT is its uni-directional nature — the model is only trained to predict the future left-to-right context.

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

[ELMo blog by Masato Hagiwara](http://www.realworldnlpbook.com/blog/improving-sentiment-analyzer-using-elmo.html)

[ULMFiT](https://arxiv.org/pdf/1801.06146.pdf)

[Fastai blog on ULMFiT](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)

[Attention is All You Need](https://arxiv.org/pdf/1511.01432)

[Semi-supervised Sequence Learning](https://arxiv.org/pdf/1511.01432)

[GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

[Byte Pair Encoding](https://arxiv.org/pdf/1508.07909)

[BERT]()

[GPT-2]()

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

