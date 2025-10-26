---
title: "Mini StarCoder2 - Data (WIP)"
tags: ["llm", "data"]
ShowToc: false
---

One of the projects I am looking forward to is training an LLM from scratch. For this project, I am thinking about creating a mini LLM version of StarCoder2 model. [StarCoder2](https://arxiv.org/pdf/2402.19173) is a CodeLLM.

To limit the scope here's a brief outline of the project

* Mini LLM, maybe 200M parameter model as a start
* Focusing only on Python as a language

The Chinchilla paper suggests using roughly 20x the number of parameters in tokens for optimal training. For my 200M parameter model, this means ~4B tokens would be ideal. To start, I'll experiment with a smaller slice of 500M tokens.

 I will document my process here as I make progress.

---

The section 2 in the paper outlines the various data sources used. I will start with a similar approach.

## Data Sources

