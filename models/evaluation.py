import csv
import numpy as np
from sklearn import metrics


def check_matrix(matrix, gold, pred):
  """Check matrix dimension."""
  if matrix.size == 1:
    tmp = matrix[0][0]
    matrix = np.zeros((2, 2))
    if (pred[1] == 0):
      if gold[1] == 0:  #true negative
        matrix[0][0] = tmp
      else:  #falsi negativi
        matrix[1][0] = tmp
    else:
      if gold[1] == 0:  #false positive
        matrix[0][1] = tmp
      else:  #true positive
        matrix[1][1] = tmp
  return matrix


def compute_f1(pred_values, gold_values):
  matrix = metrics.confusion_matrix(gold_values, pred_values)
  matrix = check_matrix(matrix, gold_values, pred_values)

  #positive label
  if matrix[0][0] == 0:
    pos_precision = 0.0
    pos_recall = 0.0
  else:
    pos_precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    pos_recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])

  if (pos_precision + pos_recall) != 0:
    pos_F1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall)
  else:
    pos_F1 = 0.0

  #negative label
  neg_matrix = [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

  if neg_matrix[0][0] == 0:
    neg_precision = 0.0
    neg_recall = 0.0
  else:
    neg_precision = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[0][1])
    neg_recall = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[1][0])

  if (neg_precision + neg_recall) != 0:
    neg_F1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
  else:
    neg_F1 = 0.0

  f1 = (pos_F1 + neg_F1) / 2
  return f1


def extract_field(truth, submission, index):
  gold = []
  guess = []
  for key, value in truth.items():
    gold.append(value[index])
    guess.append(submission[key][index])
  return gold, guess


def compute_scoreA(truth, submission):
  gold, guess = extract_field(truth, submission, 0)
  score = compute_f1(guess, gold)
  return score


def compute_scoreB(truth, submission,num_labels):
  results = []
  total_occurences = 0
  for index in range(1, num_labels): #pass the number of labels depending on the dataset and domain setup, ignore the first column which contains binary classification labels
    gold, guess = extract_field(truth, submission, index)
    f1_score = compute_f1(guess, gold)
    weight = gold.count(True)
    total_occurences += weight
    results.append(f1_score * weight)
  return sum(results) / total_occurences #if total_occurences != 0 else 0.0


def load_data(file_path):
    """
    Load data from a tab-separated file and convert labels to boolean values.
    Return a dictionary where keys are the first column values, and values are lists of boolean labels.
    
    Parameters:
    - file_path (str): Path to the file with gold labels or predictions in txt format.
    """
    data = {}
    rowsize = None

    with open(file_path) as file:
        reader = csv.reader(file, delimiter='\t')
        count = 1
        for row in reader:
            if len(row) < 2:  #ensure at least one label column is present
                raise ValueError(f'Wrong number of columns in line {count}, expected at least 2.')

            if rowsize and len(row) != rowsize:
                raise ValueError(f'Inconsistent number of columns in line {count}.')

            rowsize = len(row)
            data[row[0]] = [bool(int(x)) for x in row[1:]]
            count += 1

    return data


def evaluate_scores(gold_path, pred_path, num_labels):
    """
    Evaluate scores for Task A and Task B.
    Return a tuple containing score_a and score_b.

    Parameters:
    - gold_path (str): Path to the gold label file.
    - pred_path (str): Path to the prediction file.
    - num_labels (int): Number of labels for multi-label classification.
    """
    
    truth = load_data(gold_path)
    submission = load_data(pred_path)
    
    # Ensure submission contains all necessary keys
    for key in truth.keys():
        if key not in submission:
            raise ValueError(f'Missing element {key} in submission')
    
    #compute f1 metric for binary classification
    if num_labels == 2:
      score_a = compute_scoreA(truth, submission)
      return score_a
    
    #compute f1 for both binary classification and multi-label classifciation
    if num_labels > 2:
      score_a = compute_scoreA(truth, submission)
      score_b = compute_scoreB(truth, submission, num_labels)
      return score_a, score_b
