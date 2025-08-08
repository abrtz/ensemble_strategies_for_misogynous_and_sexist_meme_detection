import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import os
from wordcloud import WordCloud
import math
import numpy as np

def load_dataset(dataset, split_name = "training"):
    """
    Load a dataset from a JSON file and convert it into a pandas DataFrame.
    Return a pandas DataFrame containing the dataset loaded from the JSON file.

    Parameters:
    - dataset (str): The name of the dataset.
    - split_name (str): The specific split of the dataset. Default to "training".
    """

    #load dataset
    dataset_path = f"datasets/{dataset}_{split_name}.json"
    with open(dataset_path, "r", encoding="utf8") as fl:
        content = json.load(fl)
    #convert to df
    df = pd.DataFrame.from_dict(content, orient="index")

    return df


def plot_labels_distribution(df, label_columns):
    """
    Plot the distribution of multiple label columns in a given DataFrame.
    Add counts and percentages on top of the bars.
    Return None. Display bar plots of label distributions.

    Parameters:
    - df (pd.DataFrame): The dataset containing label columns.
    - label_columns (list): A list of column names to plot distributions for.
    """
    
    num_cols = len(label_columns)
    n_cols = 2
    n_rows = math.ceil(num_cols / n_cols)

    #fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 5))  #dynamic sizing
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    #make axes 1D array
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for ax, col in zip(axes, label_columns):
        label_counts = df[col].value_counts()
        total_count = len(df[col])  #get total instances of the column
        
        plot_data = label_counts.reset_index()
        plot_data.columns = [col, 'count']

        sns.barplot(data=plot_data, x=col, y='count', hue=col, ax=ax, palette="viridis", legend=False)
        
        #annotate each bar with count and percentage
        for i, count in enumerate(label_counts.values):
            percentage = (count / total_count) * 100  #calculate percentage
            ax.text(i, count + 0.5, f"{count}\n ({percentage:.1f}%)", ha="center", va="bottom")
        
        #set plot labels and title
        ax.set_xlabel(col.title(), fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        #ax.set_title(f"Distribution of {col.capitalize()} Memes", fontsize=14)
        #ax.set_xticklabels(ax.get_xticklabels())
        ax.set_ylim(0, max(label_counts.values) * 1.1) #add space on top for text

    #turn off any unused subplots
    for i in range(len(label_columns), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()


def plot_label_distribution_filtered_df(df, labels, ds_name= "MAMI", target_value="1", split="Training"):
    """
    Plot the distribution of labels in a given DataFrame with counts and percentages on top of bars.
    Counts the occurrences of "1" in each label, calculates percentages, and displays them.
    Return None. Display a bar plot with counts and percentages on top of each bar.

    Parameters:
    - df (pd.DataFrame): The dataset containing label columns.
    - labels (list): A list of column names to plot distributions for.
    - ds_name (str) : The name of the dataset to filter for misogyny or sexism.
    - target_value (str): The value to filter for under misogynous/sexist column. Default to "1".
    """

    if ds_name=="MAMI":
        cat = "misogynous"
    else:
        cat = "sexist"
    #fiter df for target_column having value "1"
    target_df = df.loc[df[cat] == target_value]

    #count occurrences of each label
    counts = [target_df[label].value_counts().get("1", 0) for label in labels]

    #plot the bar chart
    plt.figure(figsize=(6, 5))
    ax = sns.barplot(x=labels, y=counts, palette="magma")

    #annotate each bar with count and percentage
    for i, count in enumerate(counts):
        percentage = (count / len(target_df)) * 100  #calculate percentage
        ax.text(i, count + 0.5, f"{count}\n ({percentage:.1f}%)", ha="center", va="bottom", fontsize=9)

    if ds_name=="MAMI":
        if target_value == "1":
            filtering = "Misogynous"
        elif target_value == "0":
            filtering = "Non-misogynous"
    else:
        if target_value == "1":
            filtering = "Sexist"
        elif target_value == "0":
            filtering = "Non-sexist"

    plt.xlabel("Labels", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    #plt.title(f"Label Distribution for {filtering} Memes in {split} Set", fontsize=14)
    plt.xticks(rotation=90)
    plt.ylim(0, max(counts) * 1.15)  #add space on top for text

    plt.tight_layout()
    plt.show()


def analyze_label_distribution(df, label, categories, split="",dataset="MAMI"):
    """
    Analyze and plot label distribution in the given dataframe.
    
    Parameters:
    - df (pd.DataFrame): The dataset (train, val, test) to analyze.
    - label (str): The main label ('misogynous' or 'sexist').
    - categories (list): List of categories to analyze.
    - split (str): The name of the split to analyze. Default to empty string.
    - dataset (str): The name of the dataset to analyze. Default to "MAMI".
    """

    print(f"Label distribution for {dataset} {split}\n")

    #count total non-misogynous or non-sexist instances
    non_label_instances = df[label].value_counts().get("0", 0)
    print(f"Total non-{label} instances: {non_label_instances}")
    
    #count total instances for each label
    for lbl in categories:
        total = df[lbl].value_counts().get("1", 0)
        print(f"Total {lbl} instances: {total}")
    
    print()
    plot_labels_distribution(df, [label])
    
    #filter misogynous/sexist instances and count those only
    filtered_categories = [item for item in categories if item != label]  #remove misogynous/sexist category
    label_df = df.loc[df[label] == "1"]
    for lbl in filtered_categories:
        total = label_df[lbl].value_counts().get("1", 0)
        print(f"Total {lbl} instances when meme is {label}: {total}")
    
    print()
    plot_label_distribution_filtered_df(df, categories,ds_name=dataset,split=split) #include sexist/misogynous in plot
    plot_label_distribution_filtered_df(df, filtered_categories,ds_name=dataset,split=split) #also get the distribution of the categories only
    
    #filter non-misogynous/non-sexist instances and count those only
    non_label_df = df.loc[df[label] == "0"]
    print(f"Total non-{label} instances: {non_label_instances}")
    for lbl in filtered_categories:
        total = non_label_df[lbl].value_counts().get("1", 0)
        print(f"Total {lbl} instances when meme is non-{label}: {total}")

##stratified sampling

def split_multilabel_data(df, label_columns, test_size=0.1, random_state=0):
    """
    Splits a dataset into train and test sets using MultilabelStratifiedShuffleSplit.
    Return the train and test dfs with stratified sampling.

    Parameters:
    - df (pd.DataFrame): The dataset containing label columns.
    - label_columns (list): A list of column names for labels.
    - test_size (float): Proportion of test data. Defaults to 0.1.
    - random_state (int): Random seed for reproducibility. Defaults to 0.
    """

    #get instances and labels
    x = df[["meme id","meme path","meme text","meme caption","bert representation","svm representation"]]
    y = df[label_columns].astype(str)

    #as seen at https://github.com/trent-b/iterative-stratification on 12th March
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(msss.split(x, y))

    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #merge back into one df
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    return df_train, df_test


## get word cloud from dataset splits
def get_wordcloud(df, text_col, filter_col="misogynous", filter_val="1", dataset_name="MAMI", dataset_split="training"):
    """
    Generate and display a word cloud from a given text column in a dataframe according to the label at the binary level (misogynous or sexist, dependending on the dataset).

    Parameters:
    - df (pd.DataFrame): The dataframe to use.
    - text_col (str): Column containing text data to generate the word cloud from.
    - filter_col (str): Column name to filter on. Default to "misogynous".
    - filter_val (str or int): Value to filter the column by. Default to "1".
    - dataset_name (str): Dataset name to display in the word cloud title. Default to "MAMI".
    - dataset_split (str): Dataset split to display in the word cloud title. Default to "training".
    """

    filtered_df = df.loc[df[filter_col] == filter_val] #get only the instance with the positive or negative class
    text = " ".join(filtered_df[text_col]) #combine all text entries into one string

    #generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    if filter_val == "1" or filter_val == 1:
        label= filter_col
    else:
        label= "non-"+filter_col

    #plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud of {dataset_name} {dataset_split} {label} memes")
    plt.show()

def plot_binary_label_distribution(dfs, label, splits=None, ds_names=None):
    """
    Plot the distribution of a single binary label (0 and 1) across multiple datasets.

    Parameters:
    - dfs (list of pd.DataFrame): List of DataFrames for each dataset split.
    - label (str): The binary label column name.
    - splits (list of str): List of dataset split names (e.g., ["Train", "Dev", "Test"]).
    - ds_names (list of str): Optional dataset names for labeling.
    """
    if ds_names is None:
        ds_names = ["MAMI"] * len(dfs)
    if splits is None:
        splits = [f"Split {i+1}" for i in range(len(dfs))]

    palette = sns.color_palette("viridis", n_colors=len(dfs))

    #collect data
    data = []
    for df, split,color in zip(dfs, splits,palette):
        counts = df[label].value_counts().to_dict()
        total = sum(counts.values())
        for class_value in ["0", "1"]:
            count = counts.get(class_value, 0)
            percentage = (count / total * 100) if total > 0 else 0
            data.append({
                "Label": f"{f'Non-{label}' if class_value=='0' else f'{label.title()}'}",
                "Count": count,
                "Percentage": percentage,
                "Split": split,
                "LabelVal": class_value,
                "Color": color
            })

    plot_df = pd.DataFrame(data)

    #create plot
    plt.figure(figsize=(8, len(dfs) *  0.8 + 1))
    ax = sns.barplot(
        data=plot_df,
        y="Label",
        x="Count",
        hue="Split",
        palette=palette,
        width=0.8
    )

    #annotate bars
    for container in ax.containers:
        for bar in container:
            width = bar.get_width()
            x = bar.get_x() + width
            y = bar.get_y() + bar.get_height() / 2
            if width > 0:
                ax.text(x + 1, y, f"{int(width)}"
                        # ({width / sum(container.datavalues) * 100:.1f}%)",
                        ,va='center', ha='left', fontsize=12)

    ax.set_xlabel("Count", fontsize=14)
    ax.set_ylabel("Label", fontsize=14)
    ax.tick_params(axis='y', labelsize=14)  
    ax.tick_params(axis='x', labelsize=14)
    ax.legend(fontsize=14,bbox_to_anchor=(1.02, 1.0), loc='upper left', borderaxespad=0,title=ds_names[0],title_fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(right=1.5)
    plt.show()

def plot_label_distribution_filtered_df_multi(dfs, labels, ds_names=None, target_value="1", splits=None):
    """
    Plot horizontal bar charts showing the distribution of labels across multiple datasets (e.g., train/dev/test).
    
    Parameters:
    - dfs (list of pd.DataFrame): List of DataFrames for each dataset split.
    - labels (list): List of column names (labels) to plot.
    - ds_names (list of str): List of dataset names, e.g., ["MAMI", "MAMI", "MAMI"].
    - target_value (str): Target value to filter label columns (e.g., "1").
    - splits (list of str): List of dataset split names (e.g., ["Train", "Dev", "Test"]).
    """
    if ds_names is None:
        ds_names = ["MAMI"] * len(dfs)
    if splits is None:
        splits = [f"Split {i+1}" for i in range(len(dfs))]

    #colors for each split
    palette = sns.color_palette("PuRd", n_colors=len(dfs))

    #collect counts in long-form DataFrame
    data = []

    for df, ds_name, split, color in zip(dfs, ds_names, splits, palette):
        cat = "misogynous" if ds_name == "MAMI" else "sexist"
        target_df = df[df[cat] == target_value]
        total = len(target_df)
        
        for label in labels:
            count = target_df[label].value_counts().get("1", 0)
            percentage = (count / total * 100) if total > 0 else 0
            data.append({
                "Label": label.title(),
                "Count": count,
                "Percentage": percentage,
                "Split": split,
                "Color": color
            })

    plot_df = pd.DataFrame(data)
    plt.figure(figsize=(8, len(labels) * 0.8 + 1))

    ax = sns.barplot(
        data=plot_df,
        y="Label",
        x="Count",
        hue="Split",
        palette=palette,
        dodge=True,
        width=0.8
    )

    for container in ax.containers:
        for bar in container:
            width = bar.get_width()
            x = bar.get_x() + width 
            y = bar.get_y() + bar.get_height() / 2
            #get label and split to find the corresponding percentage
            label = bar.get_label()
            for row in plot_df.itertuples():
                if row.Label == bar.get_y() and row.Count == width:
                    percentage = row.Percentage
                    break
            ax.text(x + 1, y, f"{int(width)}"
                    #({percentage:.1f}%)"
                    , va='center', ha='left', fontsize=12)
   
    ax.set_xlabel("Count", fontsize=14)
    ax.set_ylabel("Label", fontsize=14)
    ax.tick_params(axis='y', labelsize=14)  
    ax.tick_params(axis='x', labelsize=14)
    ax.legend(fontsize=14,bbox_to_anchor=(1.02, 1.0), loc='upper left', borderaxespad=0,title=ds_names[0],title_fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(right=1.5)
    plt.show()
