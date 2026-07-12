## Overview
This project contains a GPT (decoder-only transformer) built from scratch in PyTorch, with experiments on dropout, model depth, and layer normalization.

## Write-up
To document and explain the process I wrote an article on medium.
Link: https://medium.com/@viggotang/what-i-learned-building-a-gpt-from-scratch-8f09129485fb

## Repo Structure
mini-gpt
├── README.MD
├── config.py
├── data
│   ├── experiment_data
│   │   ├── dropout_data
│   │   │   ├── big_dropout_10.csv
│   │   │   ├── big_dropout_25.csv
│   │   │   ├── big_dropout_5.csv
│   │   │   ├── small_dropout_10.csv
│   │   │   ├── small_dropout_25.csv
│   │   │   └── small_dropout_5.csv
│   │   ├── layer_norm_data
│   │   │   ├── no_norm.csv
│   │   │   └── norm.csv
│   │   └── transformer_blocks_data
│   │       ├── 10_blocks.csv
│   │       ├── 1_block.csv
│   │       └── 5_blocks.csv
│   └── training_data
│       └── shakespeare.txt
├── experiment-notebooks
│   ├── README.md
│   ├── dropout-experiment.ipynb
│   ├── layer-normalization-experiment.ipynb
│   └── transformer-block-experiment.ipynb
├── generate.py
├── gpt_model.pth
├── model.py
├── tokenizer.json
├── tokenizer.py
└── train.py

## How to run it 
* **Training**
  * Navigate to the train file and run it.
* **Generating new text**
  * Navigate to the generate.py and run it.
