# Exploring Ensemble Strategies for Misogynous and Sexist Meme Detection
This repository contains the code for building and evaluating ensemble strategies on two misogynous and sexist meme datasets. The datasets used in this project were the MAMI (Fersini et al., 2022) and EXIST 2024 (Plaza et al., 2024), which not be uploaded due to privacy constraints. However, they are are available for research upon request to their respective authors: https://github.com/MIND-Lab/SemEval2022-Task-5-Multimedia-Automatic-Misogyny-Identification-MAMI- and https://nlp.uned.es/exist2024/.

# Overview
This repository contains the code related to the experiments conducted for the research master's thesis project Exploring Ensemble Strategies for Misogynous and Sexist Meme Detection. This thesis was carried out by Ariana Britez with the supervision of dr. Ilia Markov.

# Project structure

```
.
├── LICENSE
├── README.md
├── data
│   └── README.md
├── datasets
│   └── README.md
├── error_analysis
│   ├── confusion_matrices.ipynb
│   ├── error_analysis.ipynb
│   ├── pearson_coefficient.ipynb
│   └── utils_error_analysis.py
├── models
│   ├── bert.ipynb
│   ├── bert_swin.ipynb
│   ├── bert_vit.ipynb
│   ├── bertweet-large-sexism-detector.ipynb
│   ├── crossdatasets_ensemble.ipynb
│   ├── crossdatasets_roberta.ipynb
│   ├── crossdatasets_roberta_swin.ipynb
│   ├── crossdatasets_style-emo-svm.ipynb
│   ├── evaluation
│   │   └── README.md
│   ├── evaluation.py
│   ├── indomain_ensemble.ipynb
│   ├── indomain_roberta.ipynb
│   ├── indomain_roberta_swin.ipynb
│   ├── indomain_style-emo-svm.ipynb
│   ├── nrc-lexicon-en.txt
│   ├── output
│   │   └── README.md
│   └── utils_classification.py
├── preprocessing
│   ├── datasets
│   │   ├── EXIST2024_test.json
│   │   ├── EXIST2024_training.json
│   │   ├── EXIST2025_test.json
│   │   ├── EXIST2025_training.json
│   │   ├── MAMI_test.json
│   │   └── MAMI_training.json
│   ├── dea.ipynb
│   ├── gold_labels.ipynb
│   ├── image_preprocessing.ipynb
│   ├── preprocessing.ipynb
│   ├── utils.py
│   └── utils_dea.py
└── requirements.txt
```

## Requirements

Experiments ran in Python 3.13.4. \
All required libraries and versions are provided in the requirements.txt file. Run the following command to install them.

`pip install -r requirements.txt`

## README

A README is provided per subfolder with instructions on how to run the code.
