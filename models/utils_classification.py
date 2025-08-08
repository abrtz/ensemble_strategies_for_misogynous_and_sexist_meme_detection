import pandas as pd
import os
import json
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV
from skmultilearn.problem_transform import BinaryRelevance

import numpy as np
from scipy.sparse import hstack

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

from evaluation import *

try:
    import google.colab
    runs_in_colab = True
except ImportError:
    runs_in_colab = False

if runs_in_colab:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from datasets import Dataset
    import torch

##preprocessing

def remove_urls(text):

    words = word_tokenize(text) #tokenize text
    url_pattern = r'\.(co|com|org|me)\b'  #define the regex pattern to match tokens with .co, .com, or .org
    words = [word for word in words if not re.search(url_pattern, word)] #remove tokens containing .co, .com, .org or .me
    
    return ' '.join(words)

##stylometric emotion features

# NLTK lemmas
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(text)])

def load_emotion_lexicon(lexicon_path="nrc-lexicon-en.txt"):
    """
    Load the NRC Emotion Lexicon from a tab-separated text file and return a dictionary mapping 
    each word to a list of associated emotions.

    Parameters:
    - lexicon_path (str, optional): Path to the NRC emotion lexicon file. Default to "nrc-lexicon-en.txt".
        The expected file format is tab-separated with three columns: [word] [emotion] [association (0 or 1)].
    """

    #load the NRC emotion lexicon into a dictionary with emotion words and corresponding associations
    emotions = {}
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                emotion_word, emotion, assoc = line.strip().split('\t')
                if assoc == '1':
                    if emotion_word in emotions:
                        emotions[emotion_word].append(emotion)
                    else:
                        emotions[emotion_word] = [emotion]  
    return emotions

def get_feats_en(upos,lemmas,emotions):
    """
    Extract stylometric and emotion-based features from a text using POS tags, function words, and NRC emotion associations.

    Parameters:
    - upos (str): A space-separated string of universal POS tags corresponding to each token in the input text.
    - lemmas (str): A space-separated string of lemmatized tokens from the input text.
    - emotions (dict): A dictionary mapping words to a list of associated emotions loaded from the NRC Emotion Lexicon.

    Return:
    A pandas Series with the following three elements:
    - pos_fw_emo (str): A space-separated representation of the text using a mix of POS tags, 
                        function words, and emotion words.
    - count (int): The number of emotion words present in the text.
    - emotion_associations (str): A space-separated list of all emotion associations found in the text.
    """
    
    # extract features:
    # - pos_fw_emo = representation of the text through POS tags, function words, and emotion words (from this representation n-grams (n=1-3) are built, see vectorize below)
    # - count = number of emotion words in a text
    # - emotion_associations = emotion associations from the NRC emotion lexicon
    
    fw_list = ['ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ'] # POS tags that correspond to function words

    pos_fw_emo = []
    count = 0
    emotion_associations = []
    for i, lemma in enumerate(lemmas.split()):
      #if lemma is found in the emotion dictionary, keep it
      if lemma.lower() in emotions:
        pos_fw_emo.append(lemma)
        count += 1
        emotion_associations.append(emotions[lemma.lower()])
      else:
        #if it is a function word, keep the lemma
        if upos.split()[i] in fw_list:
          pos_fw_emo.append(lemma)
        else:
          #if not keep the POS tag
          pos_fw_emo.append(upos.split()[i])
    emotion_associations = [emo for sublist in emotion_associations for emo in sublist]

    return pd.Series([' '.join(pos_fw_emo), count, ' '.join(emotion_associations)])

def get_stylometric_emotion_features(df, text_col="svm representation", label_col=["misogynous"]):
    """
    Process a df by getting text from given column and labels, lemmatizing text, extracting POS tags,
    and computing emotion-based features using NRC lexicon.
    Return the processed df with added features as new columns.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - text_col (str): Name of the text column. Default to "svm representation".
    - label_col (list): Name of the label column. Default to ["misogynous"].
    """

    #rename text and label columns
    df = df[[text_col] + label_col].rename(columns={text_col: "text"})
    
    #NLTK lemmatization
    df["lemmas"] = df["text"].apply(lemmatize_text)
    
    #NLTK POS tagging
    pos_text = df["text"].apply(lambda row: pos_tag(word_tokenize(row), tagset="universal"))
    df["upos"] = pos_text.apply(lambda row: " ".join([item[1] for item in row]))

    #load emotions lexicon
    emotions = load_emotion_lexicon()
    
    df[['pos_fw_emo', 'count', 'emotion_associations']] = df.apply(
        lambda x: get_feats_en(x['upos'], x['lemmas'],emotions), axis=1
    )
    
    return df

##binary classification

def create_classifier(train_intances, train_labels):
    """
    Create and train a text classification model using a SVM model. 
    The model applied is LinearSVC.
    
    Return:
    - tuple: A trained LinearSVC model and the TfidfVectorizer used for text transformation.
    
    Parameters:
    - train_instances (list of str): The training dataset consisting of text instances.
    - train_labels (list): The corresponding labels for the training data.
    """
    
    vec = TfidfVectorizer(#min_df=5, # If a token appears fewer times than this, across all documents, it will be ignored
                        #tokenizer=nltk.word_tokenize, # we use the nltk tokenizer
                        #stop_words=stopwords.words('english') # stopwords are removed
                    )
    
    X = vec.fit_transform(train_intances) #convert text data to numerical feature vectors

    model = LinearSVC(max_iter=10000,random_state=0) #initialize LinearSVC model

    model.fit(X, train_labels) #train model
    
    # filename_vectorizer = f"../../models/{modelname}_vectorizer.sav"
    # pickle.dump(vec, open(filename_vectorizer, 'wb'))
    # filename_model = f"../../models/{modelname}_classifier.sav"
    # pickle.dump(model, open(filename_model, 'wb'))
    
    return model, vec


##multi-label classification

def create_multilabel_classifier(train_intances, train_labels, strategy):
    """
    Create and train a multi-label text classification model using an SVM model 
    and the specified multi-label classification strategy.
    The model applied is LinearSVC.
    
    Return:
    - tuple: A trained LinearSVC model and the TfidfVectorizer used for text transformation.
    
    Parameters:
    - train_instances (list of str): The training dataset consisting of text instances.
    - train_labels (list): The corresponding labels for the training data.
    - strategy (skmultilearn model wrapper): A multi-label classification strategy, such as BinaryRelevance or LabelPowerset.
    """
    
    vec = TfidfVectorizer(
                    )
    
    X = vec.fit_transform(train_intances) #convert text data to numerical feature vectors
    
    #convert multi-label problem according to strategy used
    #use LinearSVC model
    model = strategy(LinearSVC(max_iter=10000,random_state=0)) 

    model.fit(X, train_labels) #train model
    
    # filename_vectorizer = f"../../models/{modelname}_vectorizer.sav"
    # pickle.dump(vec, open(filename_vectorizer, 'wb'))
    # filename_model = f"../../models/{modelname}_classifier.sav"
    # pickle.dump(model, open(filename_model, 'wb'))
    
    return model, vec


def classify_data(test_instances, model, vec):
    """
    Classify a list of test instances using a trained model and a vectorizer.

    Return:
    - numpy.ndarray: An array of predicted labels for the provided `test_instances`.

    Parameters:
    - test_instances (list of str): A list of text instances to be classified. Each element in the list is a string of text that will be processed by the model.
    - model (object): A trained SVM model to predict the labels of the test instances.
    - vec (object): A vectorizer (e.g., TfidfVectorizer, CountVectorizer) that has been fitted on training data. 
                    It will transform the `test_instances` into a feature vector representation that can be processed by the `model`.
    """

    test_vectors = vec.transform(test_instances)
    predictions = model.predict(test_vectors)

    return predictions


def create_hierarchical_multilabel_classifier(X_train,y_train_binary,X_train_bin_positive,y_train_categories,X_test,binary_label,fine_grained_labels,strategy=BinaryRelevance):
    """
    Train a binary classification model and a multi-label classification model to do hierarchical classification on test data.
    First classify instances using a binary classification model to predict if they are misogynous/sexist (binary label).
    Then, apply a fine-grained classification model to instances that were labelled as misogynous/sexist in the binary model. 

    Parameters:
    - X_train (list or array-like): The training dataset consisting of text instances.    
    - y_train_binary (list): The corresponding labels for the binary classification (misogynous/sexist).
    - X_train_bin_positive (list or array-like): The training dataset consisting of text instances that are positive ((misogynous/sexist) in gold binary label.
    - y_train_categories (list or array-like): The corresponding labels of X_train_bin_positive for the fine-grained categories for multi-label classification.
    - X_test (list or array-like): The test data to classify.
    - binary_label (str): The name of the binary label column in the DataFrame (sexist or misogynyous).
    - fine_grained_labels (list): A list of column names representing the labels for category classification.
    - strategy (skmultilearn model wrapper): A multi-label classification strategy, such as BinaryRelevance or LabelPowerset. Default to BinaryRelevance.

    Return:
    predictions in pandas df and as an array as well as trained models and vectorizers.
    - pred_df (pandas.DataFrame): DataFrame containing predictions for the test instances, with binary labels and fine-grained category labels.
    - binary_model: The trained binary classification model.
    - vec: The trained vectorizer for binary classification.
    - multilabel_model: The trained multi-label classification model.
    - vec_: The trained vectorizer for fine-grained classification.
    """

    #first build binary model to predict positive instances of memes (misogynous/sexist)
    binary_model, vec = create_classifier(X_train,y_train_binary)
    y_pred_binary = classify_data(X_test,binary_model, vec)

    #filter only misogynous/sexist instances for fine-grained classification
    X_test_positive = pd.DataFrame(X_test)[y_pred_binary == 1][0].tolist() #only predict fine-grained labels for these

    #initialize df with predictions for binary classification which will be populated with the categories
    #default all fine-grained labels to 0
    pred_df = pd.DataFrame({binary_label: y_pred_binary}) 
    pred_df[fine_grained_labels] = 0

    #build the multi-label classification model for fine-grained labels and predict fine-grained categories for positive instances
    multilabel_model,vec_ = create_multilabel_classifier(X_train_bin_positive,y_train_categories,strategy)
    y_pred_multilabel = classify_data(X_test_positive,multilabel_model,vec_) #this only applies to those that were labelled as 1 in binary

    #add fine-grained labels to the positive instances in the predictions df to get the labels for full dataset
    pred_df.loc[y_pred_binary == 1, fine_grained_labels] = y_pred_multilabel.toarray() 
    #y_pred_fine_grained = pred_df[fine_grained_labels].to_numpy() #get only the fine-grained labels as array
    
    return pred_df, binary_model, vec, multilabel_model, vec_

def classify_data_hierarchically(binary_model, binary_vec, multilabel_model, multilabel_vec, X_test,binary_label,fine_grained_labels):
    """
    Load trained model for binary and multi-label classification to do hierarchical classification on unseen data.
    First classify instances using a binary classification model to predict if they are misogynous/sexist (binary label).
    Then, apply a fine-grained classification model to instances that were labelled as misogynous/sexist in the binary model. 

    Return a pandas df containing predictions for the test instances, with binary labels and fine-grained category labels.

    Parameters:
    - binary_model: The trained binary classification model.
    - binary_vec: The trained vectorizer for binary classification.
    - multilabel_model: The trained multi-label classification model.
    - multilabel_vec: The trained vectorizer for fine-grained classification.
    - X_test (list or array-like): The test data to classify.
    - binary_label (str): The name of the binary label column in the DataFrame (sexist or misogynyous).
    - fine_grained_labels (list): A list of column names representing the labels for category classification.
    """

    #predict positive instances of memes (misogynous/sexist)
    y_pred_binary = classify_data(X_test,binary_model, binary_vec)

    #filter only misogynous/sexist instances for fine-grained classification
    X_test_positive = pd.DataFrame(X_test)[y_pred_binary == 1][0].tolist() #only predict fine-grained labels for these

    #initialize df with predictions for binary classification which will be populated with the categories
    #default all fine-grained labels to 0
    pred_df = pd.DataFrame({binary_label: y_pred_binary}) 
    pred_df[fine_grained_labels] = 0

    #predict fine-grained categories for positive instances
    y_pred_multilabel = classify_data(X_test_positive,multilabel_model,multilabel_vec) #this only applies to those that were labelled as 1 in binary

    #add fine-grained labels to the positive instances in the predictions df to get the labels for full dataset
    pred_df.loc[y_pred_binary == 1, fine_grained_labels] = y_pred_multilabel.toarray() 
    #y_pred_fine_grained = pred_df[fine_grained_labels].to_numpy() #get only the fine-grained labels as array
 
    return pred_df

def svm_hyperparameter_tuning(X_train,y_train):
    """
    Perform hyperparameter tuning for a Linear Support Vector Classifier (LinearSVC) model using GridSearchCV with 5-fold cross-validation.
    The scoring metric used is macro-averaged F1 score  and the model is set to `max_iter=30000` and `random_state=0` for reproducibility.

    Hyperparameters Tuned:
    - C: Regularization strength (values tested: [0.1, 0.5, 1, 5, 10])
    - tol: Tolerance for stopping criteria (values tested: [1e-2, 1e-3, 1e-4, 1e-5])
    - loss: Loss function (values tested: ['hinge', 'squared_hinge'])
    
    Print the best model configuration, the best parameter values and the best cross-validated macro-averaged F1 score.
    Return the best-performing LinearSVC model found during grid search.

    Parameters:
    - X_train (array): Training input samples of shape (n_samples, n_features).
    - y_train (array or list):  Target labels corresponding to the training data.
    """

    model = LinearSVC(max_iter=30000,random_state=0)

    #define the parameter grid for hyperparameter tuning
    parameters_dictionary = {'C':[0.1, 0.5, 1, 5, 10],
                             #'tol':[1e-2,1e-3,1e-4,1e-5], 
                             #'loss':['hinge', 'squared_hinge']
                            }

    #use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, 
                               parameters_dictionary, 
                               cv=5, 
                               verbose = 1,
                               scoring='f1_macro',
                               n_jobs=-1)

    grid_search.fit(X_train, y_train)

    #get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_f1 = grid_search.best_score_

    print('The best model was:', best_model)
    print('The best parameter values were:', best_params)
    print('The best f1-score was:', best_f1)

    return best_model


### emotion + stylometric features models

def create_binary_classifier_emotion(train_df,binary_label="misogynous",hpt=False, model=None):
    """
    Train a binary text classification model using a SVM model and multiple feature representations:
    stylometric and emotion-based features, and character-level n-grams and word-level n-grams from the text.
    
    Train a LinearSVC model either with default hyperparameters or through hyperparameter tuning.

    Return a tuple containing: 
    - clf_svc (sklearn.svm.LinearSVC): The trained binary classification model.
    - vec_tfidf1 (TfidfVectorizer): Vectorizer used on the 'pos_fw_emo' column.
    - vec_tfidf2 (TfidfVectorizer): Vectorizer used on the 'emotion_associations' column.
    - vec_char_ngram (CountVectorizer): Character n-gram vectorizer (4-7 chars) used on 'text' column.
    - vec_word_ngram (CountVectorizer): Word n-gram vectorizer (1-3 words) used on 'text' column.

    
    Parameters:
    - train_df (pandas.DataFrame): A DataFrame containing training instances with emotion and stylometric features 
                            as columns ('pos_fw_emo' and 'emotion_associations') as well as corresponding labels.
    - binary_label (str): The name of the column to be used as the target binary label. Default to "misogynous".
    - hpt (bool): Whether to perform hyperparameter tuning using GridSearchCV. 
                    If True, perform tuning before fitting the final model. Default to False.
    - model (estimator): Base classifier to use. Default is LinearSVC with max_iter=30000 and random_state=0 as parameters.
    """

    vec_tfidf1 = TfidfVectorizer(tokenizer=lambda x: x.split(), analyzer='word', ngram_range=(1, 1))
    vec_tfidf2 = TfidfVectorizer(tokenizer=lambda x: x.split(), analyzer='word', ngram_range=(2, 2))
    vec_char_ngram = CountVectorizer(analyzer='char', ngram_range=(3,6)) #tf bow character n-gram
    #vec_word_ngram = CountVectorizer(tokenizer=lambda x: x.split(), analyzer='word', ngram_range=(1, 3)) #tf bow word n-gram

    X_train = hstack((vec_tfidf1.fit_transform(train_df.pos_fw_emo), 
                      vec_tfidf2.fit_transform(train_df.emotion_associations),
                      vec_char_ngram.fit_transform(train_df.text),
                      #vec_word_ngram.fit_transform(train_df.text),
                      ), format='csr')
    y_train = train_df[binary_label].to_numpy()

    if hpt:
        clf_svc= svm_hyperparameter_tuning(X_train,y_train) #find best parameters and train model   
    
    else:
        if model is not None:
            clf_svc = model
        else:
            clf_svc = LinearSVC(max_iter=30000,random_state=0) #initialize LinearSVC model
        clf_svc.fit(X_train, y_train) #train model
    
    return clf_svc, vec_tfidf1, vec_tfidf2, vec_char_ngram #, vec_word_ngram


def create_multilabel_classifier_emotion(train_df,name_labels,model=LinearSVC(max_iter=30000,random_state=0),strategy=BinaryRelevance,hpt=False):
    """
    Train a multilabel text classification model using stylometric and emotion-based features
    and character-level n-grams and word-level n-grams from the text.
    A binary relevance classification strategy wraps around an SVM classifier.
    The model is trained with provided parameters or hyperparameter tuning is performed with GridSearchCV.
    If hpt is performed, print the best model configuration, the best parameter values and the best cross-validated macro-averaged F1 score.
    
    Return a tuple containing: 
    - clf_svc (skmultilearn classifier): The trained classification model.
    - vec_tfidf1 (TfidfVectorizer): Vectorizer used on the 'pos_fw_emo' column.
    - vec_tfidf2 (TfidfVectorizer): Vectorizer used on the 'emotion_associations' column.
    - vec_char_ngram (CountVectorizer): Character n-gram vectorizer (4-7 chars) used on 'text' column.
    - vec_word_ngram (CountVectorizer): Word n-gram vectorizer (1-3 words) used on 'text' column.
    
    Parameters:
    - train_df (pandas.DataFrame): A DataFrame containing training instances with emotion and stylometric features 
                            as columns ('pos_fw_emo' and 'emotion_associations') as well as corresponding labels.
    - name_labels (list): The name of the columns in `train_df` corresponding to the multilabel target labels.
    - model (estimator): Base classifier to use in the multilabel strategy. Default is LinearSVC with max_iter=10000 and random_state=0 as parameters.
    - strategy (skmultilearn model wrapper): Multilabel classification strategy (e.g., BinaryRelevance, LabelPowerset). Default is BinaryRelevance.
    - hpt (bool): Whether to perform hyperparameter tuning using GridSearchCV. 
                If True, perform tuning before fitting the final model. Default to False.
    """
    
    vec_tfidf1 = TfidfVectorizer(tokenizer=lambda x: x.split(), analyzer='word', ngram_range=(1, 1))
    vec_tfidf2 = TfidfVectorizer(tokenizer=lambda x: x.split(), analyzer='word', ngram_range=(2, 2))
    vec_char_ngram = CountVectorizer(analyzer='char', ngram_range=(3,6)) #tf bow character n-gram
    #vec_word_ngram = CountVectorizer(tokenizer=lambda x: x.split(), analyzer='word', ngram_range=(1, 3)) #tf bow word n-gram

    X_train = hstack((vec_tfidf1.fit_transform(train_df.pos_fw_emo), 
                      vec_tfidf2.fit_transform(train_df.emotion_associations),
                      vec_char_ngram.fit_transform(train_df.text),
                      #vec_word_ngram.fit_transform(train_df.text),
                      ), format='csr')
    y_train = train_df[name_labels].to_numpy()

    if hpt: 
        #define paramters for model with wrapped strategy
        parameters_dictionary = {
            "classifier__C":[0.1, 0.5, 1, 5, 10],
            #"classifier__tol":[1e-2,1e-3,1e-4,1e-5], 
            #"classifier__loss":['hinge', 'squared_hinge']
            }
        br_classifier = strategy(model)
        #find best parameters
        clf_svc = GridSearchCV(br_classifier, 
                        parameters_dictionary, 
                        cv=5,
                        verbose=1,
                        scoring='f1_macro', 
                        n_jobs=-1)

        clf_svc.fit(X_train, y_train) #train model

        #get the best model and parameters
        best_model = clf_svc.best_estimator_
        best_params = clf_svc.best_params_
        best_f1 = clf_svc.best_score_

        print('The best model was:', best_model)
        print('The best parameter values were:', best_params)
        print('The best f1-score was:', best_f1)
    
    else:
        clf_svc = strategy(model) #initialize LinearSVC model
        clf_svc.fit(X_train, y_train) #train model

    return clf_svc, vec_tfidf1, vec_tfidf2, vec_char_ngram #, vec_word_ngram


def classify_data_emotion(test_df, model, vec_tfidf1, vec_tfidf2, vec_char_ngram): #, vec_word_ngram):
    """
    Predict binary labels for input test data using a trained model and pre-fitted vectorizers.
    Take a test DataFrame with stylometric and emotion-based features, vectorize the relevant 
    columns using the provided TF-IDF vectorizers, and apply the trained classification model to generate predictions.

    Return:
    - y_pred : numpy.ndarray 
        The predicted binary labels for each instance in the test data.

    Parameters:
    - test_df (pandas.DataFrame): The input DataFrame containing test instances with emotion and stylometric features 
        as columns ('pos_fw_emo' and 'emotion_associations') 
    - model (object): A trained binary SVM model to predict the labels of the test instances.
    - vec1 (sklearn.feature_extraction.text.TfidfVectorizer): TF-IDF vectorizer fitted on the 'pos_fw_emo' column during training.
    - vec2 (sklearn.feature_extraction.text.TfidfVectorizer): TF-IDF vectorizer fitted on the 'emotion_associations' column during training.
    """

    X_test = hstack((vec_tfidf1.transform(test_df.pos_fw_emo), 
                    vec_tfidf2.transform(test_df.emotion_associations),
                    vec_char_ngram.transform(test_df.text),
                    #vec_word_ngram.transform(test_df.text),
                    ), format='csr')
    y_pred = model.predict(X_test) #make predictions on test data

    return y_pred


def create_hierarchical_multilabel_classifier_emotion(train_df,test_df,bin_label,fine_grained_labels,strategy=BinaryRelevance,bin_model=None,ml_model=None,ml_hpt=False):
    """
    Train a binary classification model and a multi-label classification model to do hierarchical classification on test data using stylometric and emotion-based features.
    First classify instances using a binary classification model to predict if they are misogynous/sexist (binary label).
    Then, apply a fine-grained classification model to instances that were labelled as misogynous/sexist in the binary model. 
    
    Parameters:
    - train_df (pandas.DataFrame): A DataFrame containing training instances with emotion and stylometric features 
                                as columns ('pos_fw_emo' and 'emotion_associations') as well as corresponding labels.
    - test_df (pandas.DataFrame): The input DataFrame containing test instances with emotion and stylometric features 
                                as columns ('pos_fw_emo' and 'emotion_associations') 
    - bin_label (str): The name of the binary label column in the DataFrame (sexist or misogynyous).
    - fine_grained_labels (list): A list of column names representing the labels for category classification.
    - strategy (skmultilearn model wrapper): A multi-label classification strategy, such as BinaryRelevance or LabelPowerset. Default to BinaryRelevance.
    - bin_model (estimator): Optional custom model (e.g., LinearSVC with specific parameters) to use for the binary classifier. 
                        If not provided, hyperparameter tuninig will be performed.
    - ml_model (estimator): Optional custom model (e.g., LinearSVC with specific parameters) to use for the multilabel classifier. 
                        If not provided, defaults to LinearSVC(max_iter=30000, random_state=0).
    - ml_hpt (bool): Whether to perform hyperparameter tuning using GridSearchCV for the multilabel classifier. 
                If True, perform tuning before fitting the final model. Default to False.

    Return:
    predictions in pandas df and as an array as well as trained models and vectorizers.
    - pred_df (pandas.DataFrame): DataFrame containing predictions for the test instances, with binary labels and fine-grained category labels.
    - binary_model: The trained binary classification model.
    - vec1: TF-IDF vectorizer used on 'pos_fw_emo' for binary classification.
    - vec2: TF-IDF vectorizer used on 'emotion_associations' for binary classification.
    - multilabel_model: The trained multilabel classifier used for fine-grained predictions.
    - vec1_: TF-IDF vectorizer used on 'pos_fw_emo' for fine-grained classification.
    - vec2_: TF-IDF vectorizer used on 'emotion_associations' for fine-grained classification.
    """

    #first build binary model to predict positive instances of memes (misogynous/sexist)
    if bin_model is not None:
        binary_model, vec_tfidf1, vec_tfidf2, vec_char_ngram = create_binary_classifier_emotion(train_df,
                                                                                                binary_label=bin_label,
                                                                                                model=bin_model)
    else: #perform hyperparameter tuning for binary classification
        binary_model, vec_tfidf1, vec_tfidf2, vec_char_ngram  = create_binary_classifier_emotion(train_df,
                                                                                                binary_label=bin_label,
                                                                                                hpt=True)
    y_pred_binary = classify_data_emotion(test_df,
                                          binary_model, 
                                          vec_tfidf1, 
                                          vec_tfidf2, 
                                          vec_char_ngram,
                                          )

    #get the texts of the positively predicted instances to only predict fine-grained labels for these
    #keep the df with the stylometric emotion features
    test_positive = test_df.copy()
    test_positive["binary_pred"] = y_pred_binary
    test_positive = test_positive[test_positive["binary_pred"] == 1]

    #initialize df with predictions for binary classification which will be populated with the categories
    #default all fine-grained labels to 0
    pred_df = pd.DataFrame({bin_label: y_pred_binary}) 
    pred_df[fine_grained_labels] = 0

    #only the instances with positive class at binary level will be use to train the fine-grained categories
    train_bin_pos = train_df[train_df[bin_label] == 1] #only the instances with positive class in binary level

    #build the multi-label classification model for fine-grained labels and predict fine-grained categories for positive instances
    if ml_model is not None:
        #use defined parameters
        multilabel_model,vec_tfidf1_, vec_tfidf2_, vec_char_ngram_ = create_multilabel_classifier_emotion(train_bin_pos,
                                                                                                        fine_grained_labels,
                                                                                                        model=ml_model,
                                                                                                        strategy=strategy)
    else:
        if ml_hpt:
            #perform hyperparameter tuning on model for fine-grained classes
            multilabel_model,vec_tfidf1_, vec_tfidf2_, vec_char_ngram_ = create_multilabel_classifier_emotion(train_bin_pos,
                                                                                                            fine_grained_labels,
                                                                                                            strategy=strategy,
                                                                                                            hpt=ml_hpt)
        else:
            #use the model with random_state=0 and max_iter=10000
            multilabel_model,vec_tfidf1_, vec_tfidf2_, vec_char_ngram_ = create_multilabel_classifier_emotion(train_bin_pos,
                                                                                                            fine_grained_labels,
                                                                                                            strategy=strategy)
    y_pred_multilabel = classify_data_emotion(test_positive,
                                              multilabel_model,
                                              vec_tfidf1_, 
                                              vec_tfidf2_, 
                                              vec_char_ngram_, 
                                              ) #this only applies to those that were labelled as 1 in binary

    #add fine-grained labels to the positive instances in the predictions df to get the labels for full dataset
    pred_df.loc[y_pred_binary == 1, fine_grained_labels] = y_pred_multilabel.toarray() 
    
    return pred_df, binary_model, vec_tfidf1, vec_tfidf2, vec_char_ngram, multilabel_model, vec_tfidf1_, vec_tfidf2_, vec_char_ngram_


def classify_data_hierarchically_emotion(binary_model, bin_vec1, bin_vec2, bin_vec3,
                                         multilabel_model, ml_vec1, ml_vec2, ml_vec3,
                                         test_df,
                                         binary_label,
                                         fine_grained_labels):
    """
    Load trained model for binary and multi-label classification to do hierarchical classification on unseen data.
    First classify instances using a binary classification model to predict if they are misogynous/sexist (binary label).
    Then, apply a fine-grained classification model to instances that were labelled as misogynous/sexist in the binary model. 
    The fine-grained model predicts a list of specific categories which can overlap, e.g., shaming, stereotype, etc.

    Return a pandas df containing predictions for the test instances, with binary labels and fine-grained category labels.

    Parameters:
    - binary_model: The trained binary classification model.
    - bin_vec1 (sklearn.feature_extraction.text.TfidfVectorizer): TF-IDF vectorizer used on 'pos_fw_emo' for binary classification.
    - bin_vec2 (sklearn.feature_extraction.text.TfidfVectorizer): TF-IDF vectorizer used on 'emotion_associations' for binary classification.
    - multilabel_model: The trained multilabel classifier used for fine-grained predictions.
    - ml_vec1 (sklearn.feature_extraction.text.TfidfVectorizer): TF-IDF vectorizer used on 'pos_fw_emo' for fine-grained classification.
    - ml_vec2 (sklearn.feature_extraction.text.TfidfVectorizer): TF-IDF vectorizer used on 'emotion_associations' for fine-grained classification.
    - test_df (pandas.DataFrame): The input DataFrame containing test instances with emotion and stylometric features 
        as columns ('pos_fw_emo' and 'emotion_associations')
    - binary_label (str): The name of the binary label column in the DataFrame (sexist or misogynyous).
    - fine_grained_labels (list): A list of column names representing the labels for category classification.
    """

    #predict positive instances of memes (misogynous/sexist)
    y_pred_binary = classify_data_emotion(test_df, 
                                          binary_model, 
                                          bin_vec1, 
                                          bin_vec2, 
                                          bin_vec3, 
                                          )

    #filter only misogynous/sexist instances for fine-grained classification
    test_positive = test_df.copy()
    test_positive["binary_pred"] = y_pred_binary
    test_positive = test_positive[test_positive["binary_pred"] == 1]

    #initialize df with predictions for binary classification which will be populated with the categories
    #default all fine-grained labels to 0
    pred_df = pd.DataFrame({binary_label: y_pred_binary}) 
    pred_df[fine_grained_labels] = 0

    #predict fine-grained categories for positive instances
    y_pred_multilabel = classify_data_emotion(test_positive,
                                              multilabel_model,
                                              ml_vec1, 
                                              ml_vec2, 
                                              ml_vec3, 
                                              ) #this only applies to those that were labelled as 1 in binary

    #add fine-grained labels to the positive instances in the predictions df to get the labels for full dataset
    pred_df.loc[y_pred_binary == 1, fine_grained_labels] = y_pred_multilabel.toarray() 
 
    return pred_df

## evaluation

def evaluate_binary_classification(gold_label_file, predictions_file, 
                                   y_true, y_pred, 
                                   label_names, 
                                   gold_labels_txt, predictions_txt,
                                   model_name="Model"):
    """
    Generate and print the classification report and confusion matrix for a given model.
    Print the evaluation report with ICM, ICM Norm and F1 from PyEvALL metrics.
    Return None.
    
    Parameters:
    - gold_label_file(str): Path to the file with gold labels in PyEvALL format.
    - predictions_file (str): Path to the file with predicted labels in PyEvALL format.
    - y_true (list or array): Gold labels
    - y_pred (list or array): Predicted labels
    - label_names (list of str): The list of label names corresponding to the binary problem.
    - gold_labels_txt (str): Path to the file with gold labels in txt format for MAMI f1 metric.
    - predictions_txt (str): Path to the file with predictions in txt format for MAMI f1 metric.
    - model_name (str): Name of the model (default: "Model")
    """

    if not isinstance(y_pred, list):
        y_pred = y_pred.tolist()

    #print classification report
    report = classification_report(y_true, y_pred,target_names=label_names)
    print(f"{'-'*100}\nClassification Report for {model_name}:\n{report}\n{'-'*100}")
    
    #confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(f"{'-'*100}\nConfusion matrix for {model_name}:")
    #display confusion matrix
    disp = ConfusionMatrixDisplay(cf_matrix,display_labels=label_names)
    #disp.plot()
    disp.plot(cmap=plt.cm.Blues)
    for text in disp.ax_.texts:
        text.set_fontsize(16)
    plt.title(f"{model_name}",fontsize=16)
    plt.show()
    print(f"{'-'*100}")

    #PyEvALL evaluation metrics
    print(f"{'-'*100}\nPyEvaLL Metrics for {model_name}:\n")
    test = PyEvALLEvaluation() 
    params = dict() 
    #params[PyEvALLUtils.PARAM_REPORT] = PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED #to get full dict report
    params[PyEvALLUtils.PARAM_REPORT] = PyEvALLUtils.PARAM_OPTION_REPORT_DATAFRAME
    metrics = ["ICM", "ICMNorm", "FMeasure"] 
    report = test.evaluate(predictions_file, gold_label_file, metrics, **params) 
    #report.print_report()
    report.print_report_tsv()
    print(f"{'-'*100}")

    #MAMI F1 metric for double-checking:
    #binary is macro-f1 and multi-label is weighted-f1
    print(f"{'-'*100}\nMAMI F1 Metrics for {model_name}:\n")
    n_labels = 2

    score_a = evaluate_scores(gold_labels_txt, predictions_txt, n_labels)
    print(f"Binary classification macro-F1 score: {score_a}")
    print(f"{'-'*100}")

##evaluate multi-label classification

def evaluate_multilabel_classification(gold_label_file, predictions_file, 
                                       y_true, y_pred, 
                                       label_names,
                                       gold_labels_txt, predictions_txt,
                                       hierarchy=True):
    """
    Evaluate the performance of a multi-label classification model 
    by printing a classification report and confusion matrices for each label.

    Parameters:
    - gold_label_file(str): Path to the file with gold labels in PyEvALL format.
    - predictions_file (str): Path to the file with predicted labels in PyEvALL format.
    - y_true (array-like): Gold binary labels (multi-label binary matrix) for the test set.
    - y_pred (array-like): Predicted binary labels (multi-label binary matrix) for the test set.
    - label_names (list of str): The list of label names corresponding to the multi-label problem.
    - gold_labels_txt (str): Path to the file with gold labels in txt format for MAMI f1 metric.
    - predictions_txt (str): Path to the file with predictions in txt format for MAMI f1 metric.
    - hierarchy (Bool): whether the evaluation considers hierarchical evaluation (first binary classification and then multi-label). Ddault to True.
    """
    
    #convert labels to numpy array
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.toarray()
    y_true = np.array(y_true)

    #get first column (binary classification)
    y_true_bin = y_true[:, 0]
    y_pred_bin = y_pred[:, 0]

    #create empty matrices to convert the first column to negative class
    y_true_neg_class = np.zeros((len(y_true), 1))
    y_pred_neg_class = np.zeros((len(y_pred), 1))

    #assign values as one-hot encoding binary classification
    y_true_neg_class[:, 0] = (y_true_bin == 0)  #negative binary class (misognynous/sexist)
    y_pred_neg_class[:, 0] = (y_pred_bin == 0)

    #get the remaining multilabel columns
    y_true_ml = y_true[:, 1:]
    y_pred_ml = y_pred[:, 1:]

    #combine the binary labels with the remaining labels
    y_true_mod = np.hstack((y_true_neg_class, y_true_ml))
    y_pred_mod = np.hstack((y_pred_neg_class, y_pred_ml))

    #adjuste label names
    #new_label_names = [f"{label_names[0]}", f"non-{label_names[0]}"] + label_names[1:]
    new_label_names = [f"non-{label_names[0]}"] + label_names[1:]

    #print classification report
    report = classification_report(y_true_mod, y_pred_mod, target_names=new_label_names, zero_division=0)
    print(f"{'-'*100}\nClassification Report:\n{report}\n{'-'*100}")
    
    #compute confusion matrices for each label
    cf_matrices = multilabel_confusion_matrix(y_true_mod, y_pred_mod)
    print(f"{'-'*100}\nConfusion matrices:")
    #print confusion matrices for each label
    for i, matrix in enumerate(cf_matrices):
        print(f"Confusion Matrix for '{new_label_names[i]}' label:")
        print(matrix)
        print()
    print(f"{'-'*100}")

    #PyEvALL evaluation metrics
    print(f"{'-'*100}\nPyEvaLL Metrics:\n")
    test = PyEvALLEvaluation() 
    params = dict() 

    if hierarchy:
        multilabel_hierarchy = {"yes": label_names, "no":[]} 
        params[PyEvALLUtils.PARAM_HIERARCHY]= multilabel_hierarchy
        metrics=["ICM", "ICMNorm" ,"FMeasure"] 
    
    elif not hierarchy:
        metrics=["FMeasure"] 
    
    #params[PyEvALLUtils.PARAM_REPORT] = PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED  #to get full dict report
    params[PyEvALLUtils.PARAM_REPORT] = PyEvALLUtils.PARAM_OPTION_REPORT_DATAFRAME
    report = test.evaluate(predictions_file, gold_label_file, metrics, **params) 
    #report.print_report()
    report.print_report_tsv()
    print(f"{'-'*100}")

    #MAMI F1 metric for double-checking:
    #binary is macro-f1 and multi-label is weighted-f1
    print(f"{'-'*100}\nMAMI F1 Metrics:\n")
    n_labels=len(label_names)

    score_a,score_b = evaluate_scores(gold_labels_txt, predictions_txt, n_labels)
    print(f"Binary classification macro-F1 score: {score_a}")
    print(f"Multi-label classification weighted-F1 score: {score_b}")
    print(f"{'-'*100}")


##convert predictions to PyEVALL format

def get_pyevall_evaluation_file_pred(df,bin_label,labels,test_case,type_eval,preds):
    """
    Convert a DataFrame of labeled meme dataset into a format suitable for PyEvALLEvaluation.

    Parameters:
    - df (pandas.DataFrame): A DataFrame containing meme id and associated labels. 
    - bin_label (str): The name of the binary label column in the DataFrame (sexist or misogynyous).
    - labels (list): A list of column names representing the labels for evaluation in the dataset.
    - test_case (str): The test case identifier to be added as a new column in the output DataFrame, e.g. "MAMI" or "EXIST2024".
    - type_eval (str): The type of evaluation format to be used. It can be:
        - `"binary"`: For binary classification.
        - `"hierarchical"`: For multi-label classification.
        - `"flat"`: For a flat multi-label classification format.
    - predictions (np.ndarray): A NumPy array containing predicted labels.

    Return a list of dictionaries formatted according to the PyEvALLEvaluation requirements including the following key:value pairs:
            - `test_case`: The name of the dataset.
            - `id`: The meme id.
            - `value`: The labels depending on the `type_eval`.
        
        The structure of the `value` column will vary depending on the evaluation type:
        - For `type_eval == "binary"`, it will contain "yes" or "no".
        - For `type_eval == "hierarchical"`, it will contain a list of label names where the value is "1".
        - For `type_eval == "flat"`, it will contain a list of "yes" or "no" values for the binary label and label names for other categories where the value is "1".
    """

    #convert files to input required by PyEvALLEvaluation
    pred_labels = df[["meme id"]].copy()
    pred_labels.insert(0, "test_case", [test_case] * (len(pred_labels)), True) #add the test_case column as per the library requirements
    
    if type_eval == "binary":
        #binary labels
        pred_labels["value"] = preds
        bin_labels = pred_labels
        bin_labels = pred_labels.replace({"value":0}, "no").replace({"value":1}, "yes") #convert values to yes and no
        bin_labels.rename(columns={"meme id": "id"}, inplace=True) #rename the id column as the requirements
        bin_labels["id"] = bin_labels["id"].astype(str)  #convert "id" column to string values
        labels_df = bin_labels

    if type_eval == "hierarchical":
        # multi-label categories hierarchically, only fine-grained unless "no" subclass
        #pred_df = pd.DataFrame(preds.toarray(), columns=labels) #convert predicted labels array to df with labels as column names
        h_multilabel_labels = pred_labels[["test_case","meme id"]].reset_index(drop=True) #keep all categories and drop index to merge with predictions df
        h_multilabel_labels = pd.concat([h_multilabel_labels, preds], axis=1) #concatenate with dataset df along columns (axis=1)
        value_cols = h_multilabel_labels.columns[2:] #filter only category columns
        h_multilabel_labels["value"] = h_multilabel_labels[value_cols].apply(
            lambda row: [
                col for col, val in row.items() if val == 1 and col != bin_label]  #if binary label is 1, keep only the fine-grained labels
                or ["no"], #if no label, assign ["no"]
                axis=1
            )
        h_multilabel_labels = h_multilabel_labels[["test_case","meme id","value"]]
        h_multilabel_labels.rename(columns={"meme id": "id"}, inplace=True) #rename the id column as the requirements
        h_multilabel_labels["id"] = h_multilabel_labels["id"].astype(str)  #convert "id" column to string values
        labels_df = h_multilabel_labels
    
    if type_eval == "flat":
        # multi-label flat (all labels)
        if not isinstance(preds, np.ndarray):
            preds = preds.toarray()
        pred_df = pd.DataFrame(preds, columns=labels) #convert predicted labels array to df with labels as column names
        flat_multilabel_labels = pred_labels[["test_case","meme id"]].reset_index(drop=True) #keep all categories and drop index to merge with predictions df
        flat_multilabel_labels = pd.concat([flat_multilabel_labels, pred_df], axis=1) #concatenate with dataset df along columns (axis=1)
        value_cols = flat_multilabel_labels.columns[2:] #filter only category columns
        flat_multilabel_labels["value"] = flat_multilabel_labels[value_cols].apply(
            lambda row: [
                "yes" if col == bin_label and val == 1
                else "no" if col == bin_label and val == 0
                else col #keep category column names where label is 1
                for col, val in row.items() if val == 1  or col == bin_label], 
                axis=1
            ) #get list of labels per instance when the value is 1. yes and no for binary label
        flat_multilabel_labels = flat_multilabel_labels[["test_case","meme id","value"]]
        flat_multilabel_labels.rename(columns={"meme id": "id"}, inplace=True) #rename the id column as the requirements
        flat_multilabel_labels["id"] = flat_multilabel_labels["id"].astype(str)  #convert "id" column to string values
        labels_df = flat_multilabel_labels

    
    labels_list = labels_df.to_dict(orient="records") #convert df into a list of dictionaries as per requirements
    
    return labels_list

#convert predictions to mami evaluation
def get_txt_evaluation_file_pred(df,type_eval,preds):
    """
    Convert a DataFrame of labeled meme dataset into a format suitable for MAMI evaluation.

    Parameters:
    - df (pandas.DataFrame): A DataFrame containing meme id and associated labels.
    - type_eval (str): The type of evaluation format to be used. It can be:
        - `"binary"`: For binary classification.
        - `"hierarchical"`: For hierarchical multi-label classification.
        - `"flat"`: For a flat multi-label classification format.

    Return a df formatted according to the MAMI evaluation metrics including the meme ID and the labels.
    """

    #convert files to input required by mami evaluation
    pred_labels = df[["meme id"]].copy()
    
    if type_eval == "binary":
        #binary evaluation
        pred_labels["value"] = preds #add binary predictions as new column
    
    else:
        if not isinstance(preds, np.ndarray):
            preds = preds.toarray()
        pred_df = pd.DataFrame(preds) #convert predictions to df
        pred_labels = pred_labels.reset_index(drop=True) #drop index to merge both dfs
        pred_labels = pd.concat([pred_labels, pred_df], axis=1)  #merge meme id and labels

    pred_labels["meme id"] = pred_labels["meme id"].astype(str)  #convert "id" column to string values
    
    return pred_labels


##save predictions for PyEvall evaluation + MAMI F1 evaluatio + csv

def write_labels_to_json(label_list,output_file,dataset_name,split_name,evaluation_name):
    """
    Write list to JSON file for PyEvALL Evaluation.
    Return None.

    Parameters:
    - label_list (list): A list of dictionaries containing the test case, meme id and labels.
    - output_path (str): Path to the output txt file.
    - dataset_name (str): Name of the dataset, e.g. MAMI or EXIST2024.
    - split_name (str): Name of the split, e.g. training, test.
    - evaluation_name (str): Type of evaluation considered: binary, flat, hierarchical.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(label_list, f, ensure_ascii=False, indent=4)
    print(f"Saved {dataset_name} {split_name} split {evaluation_name} evaluation to {output_file}")


def write_labels_to_txt(labels_df, output_path,dataset_name,split_name):
    """
    Write df to txt file for MAMI Evaluation.
    Return None.

    Parameters:
    - labels_df (pandas.DataFrame): A DataFrame containing meme id and associated labels.
    - output_path (str): Path to the output txt file.
    - dataset_name (str): Name of the dataset, e.g. MAMI or EXIST2024.
    - split_name (str): Name of the split, e.g. training, test.
    """
    labels_df.to_csv(output_path, index=False, sep='\t', header=False)
    print(f"Saved {dataset_name} {split_name} split to {output_path}")


def save_evaluation(df, pred_dir, dataset_name, split_name, evaluation_name, model_name, predictions, bin_label, labels):
    """
    Save evaluation results to a JSON file.
    Return path to the output json and txt file with predictions.

    Parameter:
    - df (pandas.DataFrame): The DataFrame containing the dataset split with meme id and labels.
    - pred_dir (str): Directory to save the predictions file.
    - dataset_name (str): Name of the dataset.
    - split_name (str): Data split (e.g., train, test, validation).
    - evaluation_name (str): Evaluation type (e.g., binary, hierarchical, flat).
    - model_name (str): Name of the model used for predictions.
    - predictions (np.ndarray): A NumPy array containing predicted labels.
    - bin_label (str): The name of the binary label column in the DataFrame (sexist or misogynyous). 
                        This will be used to create binary categories ("yes" or "no") in the output 
                        if `evaluation_name` is set to `"binary"` or `"flat"`.
    - labels (list): A list of column names representing the labels for evaluation in the dataset.
    """
    pred_dir = pred_dir+"/"+dataset_name
    #create output directory
    os.makedirs(pred_dir, exist_ok=True)

    #convert the predictions to PyEvALL format
    pred_label_list = get_pyevall_evaluation_file_pred(df,bin_label,labels,dataset_name,evaluation_name,predictions)
    #save predictions to json file
    output_file_path_json = f"{pred_dir}/{model_name}_{dataset_name}_{split_name}_{evaluation_name}.json"
    write_labels_to_json(pred_label_list,output_file_path_json,dataset_name,split_name,evaluation_name)

    #convert the predictions to MAMI F1 format
    if isinstance(predictions, pd.DataFrame): 
        predictions = predictions.to_numpy() #convert to array when predictions are passed as df
    pred_label_df = get_txt_evaluation_file_pred(df,evaluation_name,predictions)
    output_file_path_txt = f"{pred_dir}/{model_name}_{dataset_name}_{split_name}_answer.txt"
    write_labels_to_txt(pred_label_df, output_file_path_txt,dataset_name,split_name)

    return output_file_path_json,output_file_path_txt

def save_predictions_csv(test_df, predictions, column_names, output_file):
    """
    Write predictions to a CSV file by keeping the meme id and predicted label(s) as columns.
    
    Parameters:
    - test_df (pd.DataFrame): DataFrame containing the original test instances and gold labels.
    - predictions (list or np.array): Model predictions.
    - column_names (list): A list of column names representing the labels for evaluation in the dataset.
    - output_file (str): Path to the output CSV file.
    """

    pred_df = pd.DataFrame(predictions, columns=[f"{col}_prediction" for col in column_names]) #convert predicted labels array to df with labels as column names + _prediction
    pred_df.insert(loc=0, column="meme id", value=test_df["meme id"].to_numpy()) #add the meme ids
    #save updated file
    pred_df.to_csv(output_file, index=False)
    
    print(f"Predictions saved to {output_file}")

### transformers

def load_model(model_name="google-bert/bert-base-uncased",n_labels=2):
    """
    Load a model from HuggingFace.
    Return model, tokenizer and device.

    Parameters:
    - model_name (str): The name of the model to load. Default to "google-bert/bert-base-uncased".
    - n_labels (int): The number of labels for the model to predict. Default to 2.
    """

    #load pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels, torch_dtype="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device

def classify_text(text,model,tokenizer,device):
    """
    Classifiy a given text input using a pre-trained model.
    Tokenize the input text and process it through the pre-trained model to get the predicted label.
    The model is a classification model, and the input text is tokenized using a Hugging Face tokenizer.

    Return the predicted class label (integer). The class label corresponds to the index of the highest logit from the model's output.

    Parameters:
    - text (str): A string of text to be classified.
    - model (transformers.PreTrainedModel): The pre-trained classification model to use for prediction.
    - tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to encode the input text. It should match the model's tokenizer.
    - device (torch.device): The device (CPU or GPU) on which the model and tensors should be loaded.
    """
    #tokenize input text, truncation set to True since some instances are longer than the model max sequence length
    inputs = tokenizer.encode_plus(text, padding='max_length', truncation=True, return_tensors="pt").to(device)
    #extract input_ids and attention_mask from the tokenized inputs
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    #make prediction with model
    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    #get predicted label
    predicted_label = torch.argmax(output[0], dim=1).item()

    #return predicted label
    return predicted_label


def load_dataset(data,tokenizer,device,labels=None):
    """
    Load tokenized dataset for training or evaluation.
    Return tokenized dataset as Dataset class.

    Parameters:
    - data (list): A list of training instances.
    """
    input_ids = tokenizer(data, padding='max_length', truncation=True, return_tensors="pt").to(device)
    if labels is not None:
      input_ids['labels'] = labels  #adding the labels to the tokenized data
    dict_tokenized = Dataset.from_dict(input_ids)

    return dict_tokenized

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics with precison, recall, F1 and accuracy.
    Return a dictionary with evaluation metrics.

    Parameters:
    - eval_pred (tuple): A tuple containing logits and labels.
    """

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    #calculate metrics with classification report
    report = classification_report(labels,predictions,output_dict=True)

    return {
      "precision": report["macro avg"]["precision"],
      "recall": report["macro avg"]["recall"],
      "f1": report["macro avg"]["f1-score"],
      #"accuracy": report["accuracy"]
    }


##hierarchical majority voting

def hierarchical_ensemble(y_pred_ensemble):
    """
    Apply hierarchical majority voting across multiple model predictions.
    Return ensemble predictions of shape (n_labels, n_samples) where predictions in columns 1+ are zeroed out if label 0 (binary) is 0.

    Parameters:
    - y_pred_ensemble (np.ndarray): Array of shape (n_models, n_labels, n_samples),
        where each model predicts multi-label outputs for multiple samples.
    """

    #copy the array to mask fine-grained classes
    y_pred_hierarchical = y_pred_ensemble.copy()

    #for each sample, if binary (label 0) == 0, zero out other labels
    for i in range(y_pred_ensemble.shape[0]):  #iterate over samples
        if y_pred_ensemble[i, 0] == 0:
            y_pred_hierarchical[i, 1:] = 0

    return y_pred_hierarchical