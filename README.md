# LonNLU-paper
---
# LonNLU: A Benchmark and Pre-trained Transformer Models for Meiteilon NLU

This repository contains the source code, experimental resources, and supporting materials for **LonNLU**, a benchmark suite for Natural Language Understanding (NLU) in **Meiteilon (Manipuri)** written in the Bengali script.

The repository provides implementations and experimental resources for three Meiteilon-specific pre-trained transformer models:

* **LonBERT**
* **LonRoBERTa**
* **LonALBERT**

It also contains the code and resources used for evaluating these models on multiple Meiteilon NLU tasks and for conducting the reported ablation and generative-model experiments.
A 190 MB subset of the pretraining corpus is provided in the repository for research and reproducibility purposes. The complete pretraining corpus used in the study is larger and is not fully released at this time.


## Pre-trained Model Files

Due to the large size of the pre-trained checkpoints, model weights are stored externally in [Google Drive link](https://drive.google.com/drive/folders/1XDM2abHU3G6-75297jLtKWH3WXkcOWGw?usp=sharing).

## Downstream Datasets

The Downstream_Dataset directory contains the datasets used for the five downstream NLU tasks in the LonNLU benchmark:
Text Classification
Headline Classification
Semantic Textual Similarity (STS)
Natural Language Inference (NLI)
Information Retrieval

The dataset sizes, splits, annotation procedures, label definitions, and evaluation protocols are described in the paper.

## Fine_Tuning
All the multi-lingual, baseline models and Lon models code are available in the folder FINE_TUNING.

## Reproducibility
The repository provides the main code and experimental resources required to reproduce the reported experiments, including:
Pretraining code
Fine-tuning code
Model checkpoints
A 190 MB subset of the pretraining corpus
Downstream training datasets
Resources for the reported ablation and generative-model experiments







