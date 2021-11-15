# arxiv-vixra-ml
Notebooks and other python code for competing against the results of garrettgoon.com/arxiv-vixra-quiz/  

The goal of the project is to accurately assess whether a given paper is from [arXiv](https://arxiv.org) or [viXra](https://vixra.org) based on the title or abstract alone.

## Setup and Workflow

The notebooks are from [Google Colab](https://colab.research.google.com) pro GPUs and written using tools from [PyTorch Lightning](https://www.pytorchlightning.ai) and [Weights and Biases](http://wandb.ai/) for organization and hyperparameter sweeps.

## Repo Contents

The present repo consists of the following directories:
* `arxiv_vixra_model`: package in which all `torch`/`pl` architectures, classes, helper functions, etc. are defined.
* `balanced_title`: Colab notebooks for classifying papers based on their title using various architectures.
* `balanced_abstract`: Colab notebooks for classifying papers based on their abstract using various architectures.
* `data_exploration`: Colab notebooks for exploring properties of the dataset, as well as normalizing and tokenizing the text.

## Data

The data used in the preceding notebooks was gleaned from [Kaggle's arXiv Dataset](https://www.kaggle.com/Cornell-University/arxiv) and scraping from [viXra](https://vixra.org).  Planning to link to the filtered, combined dataset used in this project at a later date.
