---
title: "Mini StarCoder2 - Data (WIP)"
tags: ["llm", "data"]
ShowToc: false
---

One of the projects I am looking forward to is training an LLM from scratch. For this project, I am thinking about creating a mini LLM version of StarCoder2 model. [StarCoder2](https://arxiv.org/pdf/2402.19173) is a Code LLM.

To limit the scope here's a brief outline of the project

* Mini LLM, maybe 200M parameter model as a start
* Focusing only on Python as a language

What does Chincilla paper tells us about how many tokens will be required? 20x the parameters of the model ~ 4B tokens of data is required. To experiment, I will start with a smaller slice first -500M tokens.

The section 2 in the paper outlines the various data sources used. I will start with a similar approach and document it here as I make progress.
