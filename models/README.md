This directory contains the code to implement the component models and the ensembles.

The SVM model was implemented with stylo-metric and emotion-based features based on Markov et al. (2021). This was done locally except for the experiments involving in-domain multi-label classification on the MAMI dataset due to its size, which were run on Google Colab. The paths are commented when that was the case in the respective notebook. This model also made use of the `nrc-lexicon-en.txt` with emotion-conveying words and their association to emotions and sentiments (Mohammad and Turney, 2013).

```
├── indomain_style-emo-svm.ipynb
├── crossdatasets_style-emo-svm.ipynb
├── nrc-lexicon-en.txt
```


This folder contains the code to fine-tune and evaluate the transformer models BERT (Devlin eta., 2019), RoBERta (Liu et al., 2019), and BERTweet for sexism detection (Al-Azzawi et al., 2023) on the EXIST 2024 dataset. 

The multimodal model was based on Wang and Markov (2024a).
The multimodal models combining Swin Trasnformer V2 (Liu et al., 2022). and ViT (Dosovitskiy et al., 2021) with either BERt and RoBERTa were also implemented on the EXIST 2024 dataset for binary sexism classification in memes.

RoBERTa as well as the multimodal model combining Swin Transformer V2 and RoBERTa were then implemented across all tasks on the two datasets used in this thesis: binary and multi-label classification.
This was done both in-domain and in cross-dataset evaluation setups.

The notebook with these models were run on Google Colab. All necessary requirements are listed in each notebook.

```
├── bert.ipynb
├── bert_swin.ipynb
├── bert_vit.ipynb
├── bertweet-large-sexism-detector.ipynb
├── indomain_roberta.ipynb
├── indomain_roberta_swin.ipynb
├── crossdatasets_roberta.ipynb
├── crossdatasets_roberta_swin.ipynb
```

All functions to train or fine-tune the models are contained in the `utils_classification.py` file.
Moreover, the [script provided by Fersini et. al (2022)](https://github.com/MIND-Lab/SemEval2022-Task-5-Multimedia-Automatic-Misogyny-Identification-MAMI-/blob/main/Evaluation/evaluation.py) is also implemented to calculate the metrics following the implementation on the MAMI Shared Task (`evaluation.py`).

These functions make use of the gold labels and predictions which are stored in the evaluation directory to output the results.

```
├── utils_classification.py
├── evaluation.py
├── evaluation
│   └── README.md
```

The predictions from the component models were then combined in a hard majority voting ensemble, both in in-domain and cross-dataset settings.

```
├── indomain_ensemble.ipynb
├── crossdatasets_ensemble.ipynb
```

The final predictions of each component model as well as the ensembles were saved on the `output` directory.

```
├── output
│   └── README.md
```