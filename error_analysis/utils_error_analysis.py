from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,multilabel_confusion_matrix
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.contingency_tables import mcnemar
import random

def get_fpr_ppv(y_true, y_pred,model_name,label_names,setup="in-domain"):
    """
    Calculate the false positive rate (FPR) and positive predicted value (PPV) for a given model.
    Return None. 
    Print the FPR and PPV.

    Parameters:
    - y_true (array-like): True binary labels (0 or 1).
    - y_pred (array-like): Predicted binary labels (0 or 1) from the model.
    - model_name (str): Name of the model (used for display purposes in the output).
    - label_names (list): Name of all labels in dataset, used to calculate values from multi-label classification.
    - setup (str): Description of the evaluation setup, e.g., "in-domain" or "cross-dataset".
    """

    print(f"{model_name} - {setup}")

    if y_true.ndim == 2:
        cm = multilabel_confusion_matrix(y_true, y_pred)
        
        all_metrics = []

        for labelname, matrix in zip(label_names, cm):
            tn, fp, fn, tp = matrix.ravel()

            #false positive rate
            fpr = fp/(fp + tn)
            #positive predictive value
            ppv = tp/(tp + fp)
            all_metrics.append((fpr, ppv))

            print(f"Class {labelname:<20}| FPR: {fpr:<5.2f} | PPV: {ppv:.2f}")
        
        #average all classes
        avg_fpr = np.mean([m[0] for m in all_metrics])
        avg_ppv = np.mean([m[1] for m in all_metrics])
        print(f"Average FPR: {avg_fpr:.2f}\nAverage PPV: {avg_ppv:.2f}")

    else:
        cm = confusion_matrix(y_true, y_pred)

        #extract TP, TN, FP, FN
        tn, fp, fn, tp = cm.ravel()
        #false positive rate
        fpr = fp/(fp + tn)
        #positive predictive value
        ppv = tp/(tp + fp)
        print(f"FPR: {fpr:.2f}\nPPV: {ppv:.2f}")

def get_pearson_correlations(model_preds,label_names):
    """
    Calculate and print the Pearson correlation coefficients between all pairs of models.
    Return None. Print Pearson correlation (r) for each model pair and average if multi-label.

    Parameters:
    - model_preds (dict): A dictionary where keys are model names (str) and values are 1D arrays 
    or lists of predictions, or 2D arrays/lists of shape (n_samples, n_labels).
    - label_names (list): Name of all labels in dataset.
    """

    model_names = list(model_preds.keys())
    #get possible pair combinations of models
    pairs = list(combinations(model_names, 2))

    #calculate and Pearson correlation for each pair
    for model1, model2 in pairs:
        x = model_preds[model1]
        y = model_preds[model2]

        if x.ndim == 1: #binary classifcation
            r, p_value = pearsonr(x, y)
            print(f"{model1:<15} vs {model2:<15} | Pearson r: {r:.2f}")
        
        else: #multilabel classification
            r_values = []

            print(f"{model1} vs {model2}")
            for i in range(1,len(label_names)): #ignore the binary label
                r, _ = pearsonr(x[:, i], y[:, i])
                r_values.append(r)
                label = label_names[i]
                print(f"  Label {label:<20} | Pearson r: {r:.2f}")
            
            avg_r = np.mean(r_values)
            print(f"  {'Average all':<25} | Pearson r: {avg_r:.2f}\n")
        
        print("-"*30)


### based on https://www.askpython.com/python/examples/mcnemars-test:
def get_table_mcnemar_r(y_true,preds_A,preds_B):
    """
    Compute the McNemar test for paired model predictions to determine if there is a significant
    difference in predictive performance between the two models.
    Return result (object containing McNemar test statistic and p-value).

    Parameters:
    - y_true (array-like): Ground truth labels, shape (n_samples,).
    - preds_A (array-like): Predictions from model A, shape (n_samples,).
    - preds_B (array-like): Predictions from model B, shape (n_samples,).
    """

    both_correct = 0
    A_correct_B_wrong = 0
    A_wrong_B_correct = 0
    both_wrong = 0

    #compare predictions
    for yt, pa, pb in zip(y_true, preds_A, preds_B):
        a_correct = pa == yt #compare true and prediction from model A
        b_correct = pb == yt #compare true and prediction from model B

        if a_correct and b_correct:
            both_correct += 1
        elif a_correct and not b_correct:
            A_correct_B_wrong += 1
        elif not a_correct and b_correct:
            A_wrong_B_correct += 1
        else:
            both_wrong += 1

    #create 2x2 contingency table
    table = np.array([
            [both_correct, A_correct_B_wrong],
            [A_wrong_B_correct, both_wrong]
        ])

    #calculate McNemar statistic
    result = mcnemar(table, exact=False, correction=True)  # or exact=False for large samples

    return result

def get_mcnemar_significance(y_true,model_preds,label_names):
    """
    Perform the McNemar significance test between an ensemble model and the best indvidual model for
    binary or multilabel classification to evaluate whether there is a statistically significant 
    difference in their predictions compared to ground truth. 
    In the multilabel case, this is done label-by-label, and summary statistics are printed.

    Return None. Print:
        - For each model vs. "Ensemble", print the McNemar test statistic and p-value and significance difference.
        - For multilabel: Individual label test results, average statistic and p-value across labels, and 
        number of labels with significant differences (p < 0.05).

    Parameters:
    - y_true (array-like): Ground truth labels. Shape (n_samples,) for binary classification
                             or (n_samples, n_labels) for multilabel classification.
    - model_preds (dict): Dictionary mapping model names to their predictions.
                            Each value should be an array-like object with shape matching y_true.
    - label_names (list of str): List of label names for multilabel classification.
    """

    pairs = [("Ensemble", other) for other in model_preds.keys() if other != "Ensemble"]

    for model1, model2 in pairs:

        preds_A = model_preds[model1]
        preds_B = model_preds[model2]

        print(f"{model1} vs {model2}")
        
        if y_true.ndim == 1: #binary classification

            result = get_table_mcnemar_r(y_true, preds_A, preds_B)

            print("\nMcNemar Test:")
            print(f"McNemar Statistic: {result.statistic:.2f}")
            print(f"p-value: {result.pvalue:.4f}")

            #intepretation based on p-value
            if result.pvalue < 0.05:
                print("Significant difference: yes")
            else:
                print("Significant difference: no")
        
        else: #multilabel classification
            
            stats = []
            pvals = []
            sig_labels = 0

            for i, label in enumerate(label_names):
                result = get_table_mcnemar_r(y_true[:, i], preds_A[:, i], preds_B[:, i])
                stats.append(result.statistic)
                pvals.append(result.pvalue)

                significance = "yes" if result.pvalue < 0.05 else "no"
                if result.pvalue < 0.05:
                    sig_labels += 1

                print(f"  Label {label:<20} | Stat: {result.statistic:<5.2f} | p-value: {result.pvalue:.4f} | Significant difference: {significance}")

            avg_stat = np.mean(stats)
            avg_pval = np.mean(pvals)
            avg_significance = "yes" if avg_pval < 0.05 else "no"

            print(f"\n  {'Average statistic':<24} | {avg_stat:.4f}")
            print(f"  {'Average p-value':<24} | {avg_pval:.4f}")
            print(f"  {'Significant labels (<0.05)':<24} | {sig_labels} / {len(label_names)}")
            print(f"  {'Average significant difference':<24} {avg_significance}")
    
        print("-"*30)


def get_avg_confusion_matrices(y_true, y_pred,label_names,model_name,setup="in-domain"):
    """
    Plot confusion matrices per class and an average of all classes for multi-label classficiation.

    Parameters:
    - y_true (array-like of shape (n_samples, n_classes)): True binary labels for each class in multi-label format.
    - y_pred (array-like of shape (n_samples, n_classes)): Predicted binary labels for each class in multi-label format.
    - label_names (list): Name of all labels in dataset to display in per-class plot title.
    - model_name (str): Name of the model to display in average plot title.
    """

    if setup != "in-domain":
        model_name = model_name + " (Cross-Dataset)"
    
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
    new_label_names = [f"non-{label_names[0]}"] + label_names[1:]

    cm = multilabel_confusion_matrix(y_true_mod, y_pred_mod)
    n_classes = len(new_label_names)

    n_cols = min(n_classes, 6)  #columns per row
    n_rows = int(np.ceil(n_classes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    for i, (labelname, matrix) in enumerate(zip(new_label_names, cm)):
        #per-class cf
        disp = ConfusionMatrixDisplay(matrix)
        disp.plot(ax=axes[i], cmap=plt.cm.Purples)
        for text in disp.ax_.texts:
            text.set_fontsize(16)
        axes[i].set_title(f"{labelname}", fontsize=16)
    
    plt.tight_layout()
    plt.suptitle(model_name, fontsize=18, y=1.02)
    plt.show()

    #average cf
    avg_cm = np.mean(cm, axis=0)
    disp = ConfusionMatrixDisplay(avg_cm)
    disp.plot(cmap=plt.cm.Purples, values_format=".1f")
    for text in disp.ax_.texts:
        text.set_fontsize(16)
    plt.title(f"{model_name} (avg)", fontsize=16)
    plt.show()


def get_fp_fn_per_label(test_df, y_pred, label_names, max_errors=30, seed=42):
    """
    Get false positives and false negatives for the label(s). 
    Return None. Print the meme ID, meme text, and meme caption for each error.

    Parameters:
    - test_df (pandas.DataFrame): A DataFrame containing meme id, meme text, meme captions and associated gold labels.
    - y_pred (array-like): Predicted labels.
    - label_names (str or list): Name of labels to output FP and FN.
    - max_errors (int): The maximum number of errors to print per class.
    - seed (int): Random seed for reproducibility.
    """

    random.seed(seed)

    y_true = test_df[label_names].to_numpy()
    meme_ids = test_df['meme id'].to_numpy()
    meme_text = test_df['meme text'].to_numpy()
    meme_caption = test_df['meme caption'].to_numpy()

    if y_true.ndim == 1:
        label = label_names

        false_positives = [
            {'meme id': meme_ids[i], 'meme text': meme_text[i], 'meme caption': meme_caption[i],
             'gold label': y_true[i], 'predicted': y_pred[i]}
            for i in range(len(y_true)) if y_pred[i] == 1 and y_true[i] == 0
        ]

        false_negatives = [
            {'meme id': meme_ids[i], 'meme text': meme_text[i], 'meme caption': meme_caption[i],
             'true_label': y_true[i], 'predicted': y_pred[i]}
            for i in range(len(y_true)) if y_pred[i] == 0 and y_true[i] == 1
        ]

        # Sample randomly
        false_positives = random.sample(false_positives, min(max_errors, len(false_positives)))
        false_negatives = random.sample(false_negatives, min(max_errors, len(false_negatives)))

        print("-" * 60)
        print(f"\n=== Label {label} ===")

        print(f"False Positives:\n")
        for fp in false_positives:
            for k, v in fp.items():
                print(f"{k}: {v}")
            print()
        print("-" * 60)
        print(f"\nFalse Negatives:\n")
        for fn in false_negatives:
            for k, v in fn.items():
                print(f"{k}: {v}")
            print()
        print("-" * 60)

    else:
        # Multilabel logic (apply same sampling logic inside loop)
        n_classes = y_true.shape[1]

        for class_idx in range(1, n_classes):
            class_fp = []
            class_fn = []

            for i in range(len(y_true)):
                true = y_true[i, class_idx]
                pred = y_pred[i, class_idx]

                if pred == 1 and true == 0:
                    class_fp.append({'meme id': meme_ids[i], 'meme text': meme_text[i], 'meme caption': meme_caption[i],
                                     'gold label': true, 'predicted': pred})

                elif pred == 0 and true == 1:
                    class_fn.append({'meme id': meme_ids[i], 'meme text': meme_text[i], 'meme caption': meme_caption[i],
                                     'gold label': true, 'predicted': pred})

            class_fp = random.sample(class_fp, min(max_errors, len(class_fp)))
            class_fn = random.sample(class_fn, min(max_errors, len(class_fn)))

            print("-" * 60)
            print(f"\n=== Label {label_names[class_idx]} ===")

            print("False Positives:")
            for item in class_fp:
                for k, v in item.items():
                    print(f"{k}: {v}")
                print()
            print("-" * 60)
            print("False Negatives:")
            for item in class_fn:
                for k, v in item.items():
                    print(f"{k}: {v}")
                print()
            print("-" * 60)