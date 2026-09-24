# LonNLU-paper
---
# LonNLU: A Benchmark and Pre-trained Transformer Models for Meiteilon NLU

This repository contains the source code, experimental resources, and supporting materials for **LonNLU**, a benchmark suite for Natural Language Understanding (NLU) in **Meiteilon (Manipuri)** written in the Bengali script.

The repository provides implementations and experimental resources for three Meiteilon-specific pre-trained transformer models:

* **LonBERT**
* **LonRoBERTa**
* **LonALBERT**

It also contains the code and resources used for evaluating these models on multiple Meiteilon NLU tasks and for conducting the reported ablation and generative-model experiments.


## Pre-trained Model Files

Due to the large size of the pre-trained checkpoints, model weights are stored directly in this [Google Drive link](https://drive.google.com/drive/folders/1XDM2abHU3G6-75297jLtKWH3WXkcOWGw?usp=sharing).

## Downstream Datasets

The Downstream_Dataset directory contains the datasets used for the five downstream NLU tasks in the LonNLU benchmark. The files are organized by task as follows:

 Text Classification ----`train_content_dataset.json'....Meiteilon text classification data
 Headline Classification----'News_headlineclassification_train.csv'---Meiteilon headline classification data   
 Natural Language Inference--- `train_nli1.csv` ---       Premise--hypothesis pairs with NLI labels       
 Semantic Textual Similarity--`semanticsimilarity_train.csv`....Meiteilon sentence pairs with similarity scores 
  Retrieval           ----         `retrieval_train.json`----  Query, positive passage, and negative passages  

The dataset sizes, splits, annotation procedures, label definitions, and evaluation protocols are described in the paper and the accompanying Data Card.

##Fine_Tuning
All the multi-lingual, baseline models and Lon models code are available in the folder FINE_TUNING.







