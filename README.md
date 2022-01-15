# arxiv-vixra-ml
Notebooks and other python code for competing against the results of [garrettgoon.com/arxiv-vixra-quiz](https://garrettgoon.com/arxiv-vixra-quiz/).

The goal of the project is to accurately assess whether a given paper is from [arXiv](https://arxiv.org) or [viXra](https://vixra.org) based on the title or abstract alone.

## Data

The arXiv data used in the preceding notebooks was gleaned from [Kaggle's arXiv Dataset](https://www.kaggle.com/Cornell-University/arxiv), while the viXra data was scraped from [viXra](https://vixra.org) and [can be downloaded here](https://www.dropbox.com/s/4gw9wv90kqyeh95/vixra_raw.feather?dl=0) as an 18MB `.feather` file.


## Setup and Workflow

The notebooks are from [Google Colab](https://colab.research.google.com) pro GPUs and written using tools from [PyTorch Lightning](https://www.pytorchlightning.ai) and [Weights and Biases](http://wandb.ai/) for organization and hyperparameter sweeps. Notebooks are written to read from Google Drive, so require alternations to run locally.

## Repo Contents

The present repo consists of the following directories:
* `arxiv_vixra_models`: `python` package in which all `torch`/`pl` architectures, classes, helper functions, etc. are defined.
* `data_exploration`: Colab notebooks for exploring properties of the dataset, as well as filtering, normalizing, and tokenizing the text.
* `figures`: Various figures generated from notebooks.
* `final_models`: `.pt` `state_dict`s, `.onnx` files, etc. for the best-performing models of various classes.
* `simple_baselines`: Simple baseline models against which to compare.
* `simple_recurrent`: Basic `RNN`/`LSTM`/`GRU` models, either at character level or using word-embeddings.
