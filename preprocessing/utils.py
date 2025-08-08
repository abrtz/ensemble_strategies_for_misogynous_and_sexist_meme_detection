import os
import json
import re
import pandas as pd

import nltk
from nltk import ngrams

try:
    import google.colab
    runs_in_colab = True
except ImportError:
    runs_in_colab = False

if runs_in_colab:
    import transformers
    from transformers import AutoProcessor, Blip2ForConditionalGeneration, Blip2Processor
    import torch
    import requests
    from PIL import Image
    from tqdm import tqdm


def preprocess_exist(file_path,datasplit="training"):
    """
    Preprocess a JSON file to extract information of memes for a specified data split.

    Parameters:
    - file_path (str): Path to the JSON file containing the meme data.
    - datasplit (str), optional: Specify the data split to process. Defaults to "training". Options are "training" or "test".

    Returns:
    - dict: A dictionary containing
        - "meme id" (list): A list of meme IDs corresponding to the specified data split.
        - "meme text" (list): A list of texts from each meme corresponding to the specified data split.
        - "meme path" (list): A list of relative paths where the memes are stored.
    """

    dataset_dict = {}

    filesplit = "training" if datasplit == "training" else "test"

    with open(file_path, "r", encoding="utf8") as fl:
        content = json.load(fl)

        #determine split type
        file_split = "TRAIN-MEME_EN" if datasplit == "training" else "TEST-MEME_EN"

        for v in content.values():
            if v["split"] == file_split: #only get the EN memes info
                meme_id = v["id_EXIST"] #get meme ID
                text = v["text"] #get text
                meme_path = filesplit + "/" + v["path_memes"]

                dataset_dict[meme_id] = {
                    "meme id": meme_id, 
                    "meme text": text, 
                    "meme path": meme_path
                }
        
    return dataset_dict


def preprocess_mami(file_path,datasplit="training"):
    """
    Preprocess a csv file to extract information of memes for a specified data split.

    Parameters:
    - file_path (str): Path to the csv file containing the meme data.
    - datasplit (str), optional: Specify the data split to process. Defaults to "training". Options are "training" or "test".

    Returns:
    - dict: A dictionary containing
        - "meme id" (list): A list of meme IDs corresponding to the specified data split.
        - "meme text" (list): A list of texts from each meme corresponding to the specified data split.
        - "meme path" (list): A list of relative paths where the memes are stored.
    """

    dataset_dict = {}

    filesplit = "training" if datasplit == "training" else "test"

    with open(file_path,"r",encoding="utf-8-sig") as file:
        lines = [line for line in file.read().split('\n') if line.strip()]  #exclude empty lines
        for row in lines[1:]:
            columns = row.strip("\n").split("\t")
            meme_id = columns[0] #get meme ID
            text = columns[-1] #get the text
            meme_path = filesplit + "/" + meme_id

            dataset_dict[meme_id] = {
                    "meme id": meme_id, 
                    "meme text": text, 
                    "meme path": meme_path
                }

    return dataset_dict


def write_to_json(json_dict, filename):
    """
    Write the datasets meme information to a JSON file.
    
    Parameters:
    -json_dict (dict): dictionary containing information about the datasets memes.
    -filename (str): name of the JSON file to write to.
    """
    
    with open(filename, 'w',encoding='utf-8') as outfile:
        json.dump(json_dict, outfile, ensure_ascii=False, indent=4) 
    print(f'{filename} is saved')


def get_image_text(file_path, vl_model="Salesforce/blip2-flan-t5-xl-coco", limit=None):
    """
    Extract text descriptions (captions) from images of memes stored in specified datasets and add them to the JSON data.
    Call the BLIP-2 model for image captioning.

    Parameters:
    - file_path (str): The path to the JSON file containing info about the meme datasets.
    - vl_model (str): the vision language model to do image captioning. Defaults to Salesforce/blip2-flan-t5-xl-coco model.
    - limit (int): the amount of memes to process per dataset split. Defaults to None - i.e., all memes are processed.

    Return:
    - dict: The updated JSON content dictionary with added "meme_captions" keys for each split, containing 
    the generated captions for the memes in that split.
    """

    #initialize model (as seen at https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BLIP-2/Chat_with_BLIP_2.ipynb on Jan 13th 2025)
    processor = Blip2Processor.from_pretrained(vl_model)
    model = Blip2ForConditionalGeneration.from_pretrained(vl_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #use GPU if available, otherwise use CPU
    model.to(device)

    prompt = "ignore text on the image. a photo of"
    #prompt = "a photo of"
    
    #these did not work:
    #prompt = "Question: Ignore text on the image. Describe only the visual elements including characters, objects, setting, and explicit actions. Be specific. Answer:"
    #prompt = "Question: Describe the image, including characters, objects, actions, and setting. Ignore any text. Answer:"
    #prompt = "Question: describe the image by including characters, objects, actions, and setting in the image, ignore the text on the image. Answer:"
    #prompt = "Question: describe the characters, objects, and setting in the image, ignore the text on the image. Answer:"
    #prompt = "Question: describe the characters, objects, and setting in the image, do not caption the text on the image. Answer:"
    #prompt = "Question: describe the characters, objects, and setting in the image, do not transcribe or reference the text. Answer:"
    #prompt = "Question: do not transcribe or reference the text. describe the characters, objects, and setting in the image. this is an image of... . Answer:"

    with open(file_path, "r", encoding="utf8") as fl:
        content = json.load(fl)

    for dataset in content.values():
        for data, split in dataset.items():
            if data == "MAMI":
                path = "/content/drive/MyDrive/Colab/thesis/datasets/MAMI DATASET"
            elif data == "EXIST2024":
                path = "/content/drive/MyDrive/Colab/thesis/datasets/EXIST 2024"
          
            for split_name, memes in split.items():          
                for i, meme_info in enumerate(tqdm(memes.values(), desc=f"Processing {data} {split_name}", unit="image")): #add a progress bar
                    meme_path = meme_info["meme path"] 
                    if limit is not None and i >= limit:  ##to test if the code works
                        break 
                    meme_path = os.path.join(path, meme_path) #get the complete path to each meme
                    
                    image = Image.open(meme_path)
                    
                    if image.mode != "RGB": #convert to right mode if necessary
                        image = image.convert("RGB")

                    #image captioninig
                    inputs = processor(image, text=prompt, return_tensors="pt").to(device)

                    generated_ids = model.generate(**inputs)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                    meme_caption =  generated_text.replace('\n', ' ').strip()  #trim caption

                    #add meme_captions to each meme dictionary
                    meme_info["meme caption"] = meme_caption
        
            print(f"Split {data} {split_name} completed")

    return content #return the updated content dictionary


### post-processsing image captions:

def remove_text_from_image_captions(content):
    """
    Remove the meme text from the image caption if it matches one of the phrases signalling the presence of text.

    Return the cleaned json dictionary without the text from the image and a dictionary containing the amount of meme captions cleaned with each phrase.

    Parameters:
    - content (dict): the full datasets with meme info per splits.
    """

    #dict of phrases to detect cases that contain meme text
    text_phrases = {
                "with the words":0,
                "with the caption":0,
                "with a caption":0,
                "with the text":0,
                "with a text":0,
                "with a quote":0,
                "with a message":0,
                "with text saying":0,
                "with text that says":0,
                #"and a caption that says":0,
                #"and a caption saying":0,
                #"and a caption says":0,
                "and a caption":0,
                "and a sign that says":0,
                "and a quote that says":0,
                "and a text that says":0,
                "and a text saying":0,
                "and text that says":0,
                "and the caption says":0,
                "and saying":0,
                "with a funny meme":0,
                #"saying":0,
                }
    
    #words/phrases that might be hanging at the end of the caption after cleaning
    end_phrases = [" and", "what is", " and has", " with a", " and a"]

    for dataset in content.values():
          for split in dataset.values():
               for memes in split.values():

                    for meme_info in memes.values():
                        meme_caption = meme_info["meme caption"].lower()
                        
                        #remove text from meme present in image captions
                        for phrase in text_phrases.keys():
                            if phrase in meme_caption:
                                text_phrases[phrase] += 1 #keep count of the instances that were removed for each type of phrase
                                meme_caption = meme_caption.split(phrase,1)[0].strip()
                                break #stop checking once a match is found
                                
                        #check if any of the hanging phrases appear at the end
                        pattern = r"(?:\s*(" + "|".join(map(re.escape, end_phrases)) + r")\s*)+$"
                        
                        meme_caption = re.sub(pattern, "", meme_caption).strip().rstrip(",.") #remove hanging phrases from the end of caption

                        meme_info["meme caption"] = meme_caption #update the dictionary with cleaned caption

    return content, text_phrases


def get_instances_to_reprocess(data_with_captions,n=3):
    """
    Get the meme ID of those that need to be processed again: 
    those containing at least three consecutive words from the meme text, 
    those that are empty after cleaning, those that contain the word "meme" in the caption,
    those that are "a text" or contain "a text message".

    Return a set of meme IDs for which the captions have to be redone.

    Parameters:
    - data_with_captions (str): the full datasets with meme info per splits.
    - n (int): the degree of the ngrams to check the consecutive words overlapping in text and image caption. Defaults to 3.
    """

    #empty set to store the meme IDs
    redo_memes = {"dataset": {}}
    
    for dataset in data_with_captions.values():
        for dataset_name, split in dataset.items():
            for split_name, memes in split.items():

                for meme_info in memes.values():
                    meme_caption = meme_info["meme caption"]
                    meme_text = meme_info["meme text"]
                    meme_id = meme_info["meme id"]
                    meme_path = meme_info["meme path"]
                                    
                    trigrams_meme_text = ngrams(meme_text.split(), n)

                    for grams in trigrams_meme_text:
                        trigram = ' '.join(str(i) for i in grams)
                        if trigram in meme_caption:  #if trigram in meme caption, then keep the meme_id for re-do image captioning 
                            if dataset_name not in redo_memes["dataset"]:
                                redo_memes["dataset"][dataset_name] = {}
                            if split_name not in redo_memes["dataset"][dataset_name]:
                                redo_memes["dataset"][dataset_name][split_name] = {}
                            redo_memes["dataset"][dataset_name][split_name][meme_id] = {
                                                                                    "meme id": meme_id, 
                                                                                    "meme text": meme_text, 
                                                                                    "meme path": meme_path
                                                                            }                    
                        
                    #after this post-processing, some empty captions are left as well as others describes as memes or still containing the word "text" when it is not in the image
                    if (len(meme_caption) == 0) or ("meme" in meme_caption): 
                        if dataset_name not in redo_memes["dataset"]:
                            redo_memes["dataset"][dataset_name] = {}
                        if split_name not in redo_memes["dataset"][dataset_name]:
                            redo_memes["dataset"][dataset_name][split_name] = {}
                        redo_memes["dataset"][dataset_name][split_name][meme_id] = {
                                                                            "meme id": meme_id, 
                                                                            "meme text": meme_text, 
                                                                            "meme path": meme_path
                                                                        }

    return redo_memes

def replace_cleaned_data_with_updated_captions(file_after_cleaning, redone_memes):
    """
    Replace the meme captions of the memes that were redone after post-processing in the previously cleaned file.

    Return a cleaned dict with new meme captions.

    Parameters:
    - file_after_cleaninig (str): path to previously cleaned json file with image captions.
    - redone_memes (dict): the subset of data for which meme captions were re-processed.
    """

    with open(file_after_cleaning, "r", encoding="utf8") as fl:
        previous_cleaned_content = json.load(fl)

    for cleaned_dataset, redone_dataset in zip(previous_cleaned_content.values(), redone_memes.values()):
        for cleaned_ds, redone_ds in zip(cleaned_dataset.values(), redone_dataset.values()):
            for cl_memes, rd_memes in zip(cleaned_ds.values(), redone_ds.values()):
                for rd_meme_key, rd_meme_info in rd_memes.items():  #iterate over new meme data
                    if rd_meme_key in cl_memes:  #if meme exists in cleaned dataset
                        cl_memes[rd_meme_key] = rd_meme_info  #update it
    
    return previous_cleaned_content


def updated_and_write_cleaned_data(redone_captions, prev_file_after_cleaninig, path_to_cleaned_data):
    """
    Remove the text from the image captions and update the previously cleaned file with the new captions.
    Save a new json file.
    
    Return the updated dict with meme info and a dictionary containing the amount of meme captions cleaned with each phrase signalling the presence of meme text.

    Parameters:
    - redone_captions (str): path to the subset of data for which meme captions were re-processed.
    - prev_file_after_cleaninig (str): path to previously cleaned json file with image captions.
    - path_to_cleaned_data (str): path to json file where cleaned data will be saved.
    - path_to_captions_to_redo (str): path to json file where memes to redo will be saved.
    - n (int): the degree of the ngrams to check the consecutive words overlapping in text and image caption. Defaults to 3.
    """

    with open(redone_captions, "r", encoding="utf8") as fl:
        redone_memes = json.load(fl)
      
    cleaned_data, removed_phrases = remove_text_from_image_captions(redone_memes) #get rid of text from meme text
    #cleaned_data = clean_meme_words(cleaned_data) #get rid of meme words when applicable
    new_cleaned_dataset = replace_cleaned_data_with_updated_captions(prev_file_after_cleaninig, cleaned_data) #replace the old cleaned dataset with the new image captions
    print("Saving data with cleaned image captions...")
    write_to_json(new_cleaned_dataset,path_to_cleaned_data)

    return new_cleaned_dataset, removed_phrases


def get_cleaned_data_and_instances_to_reprocess(redone_captions, prev_file_after_cleaninig, path_to_cleaned_data, path_to_captions_to_redo, n=3):
    """
    Remove the text from the image captions and update the previously cleaned file with the new captions.
    Save a new json file.
    For those that still contain text from meme text by checking a number of ngrams, 
    those that are empty or are described like "memes", save a new json file to reprocess image captions.

    Return a dictionary containing the amount of meme captions cleaned with each phrase signalling the presence of meme text.

    Parameters:
    - redone_captions (str): path to the subset of data for which meme captions were re-processed.
    - prev_file_after_cleaninig (str): path to previously cleaned json file with image captions.
    - path_to_cleaned_data (str): path to json file where cleaned data will be saved.
    - path_to_captions_to_redo (str): path to json file where memes to redo will be saved.
    - n (int): the degree of the ngrams to check the consecutive words overlapping in text and image caption. Defaults to 3.
    """
          
    new_cleaned_dataset, removed_phrases = updated_and_write_cleaned_data(redone_captions, prev_file_after_cleaninig, path_to_cleaned_data)
    
    print()

    memes_to_redo = get_instances_to_reprocess(new_cleaned_dataset,n)
    print("Saving data to redo image captions...")
    write_to_json(memes_to_redo,path_to_captions_to_redo)

    return removed_phrases


def print_removed_phrases(removed_phrases):
    """
    Print the total number of captions containing text from a meme,
    along with a breakdown of each removed phrase and its count.
    
    Parameter:
    - removed_phrases (dict): Dictionary with phrases as keys and counts as values.
    """
    print(f"Total captions that contained text from meme: {sum(removed_phrases.values())}")
    print()
    for phrase, total in removed_phrases.items():
        print(f"Phrase: {phrase} \t\t Total: {total}")


### after image captioning is done:

def generate_text_caption_representations(data_with_captions):
    """
    Process a JSON file containing meme data, generating representations for BERT and SVM models 
    by combining meme text with captions. The representations are added back to the data structure.

    Parameters:
    - data_with_captions (str): Path to a JSON file containing meme data. 

    Return:
    - dict: Updated data structure with added representations for each meme as:
            - "text + caption (bert)": string formatted for BERT model: "text [SEP] caption".
            - "text + caption (svm)": string formatted for SVM model: "text. caption".
    """
    
    with open(data_with_captions, "r", encoding="utf8") as fl:
        content = json.load(fl)

    for dataset in content.values():
        for split in dataset.values():
            for memes in split.values():
                for meme_info in memes.values():
                    meme_text = meme_info["meme text"].strip()
                    meme_caption = meme_info["meme caption"]

                    #create the representation for each type of model
                    bert_representation = meme_text + " [SEP] " + meme_caption
                    svm_representation = meme_text + ". " + meme_caption

                    #add the representations to the dictionary
                    meme_info["text + caption (bert)"] = bert_representation
                    meme_info["text + caption (svm)"] = svm_representation

    return content


def process_and_save_splits(complete_datasets):
    """
    Process the dataset to create a dictionary for each split with meme info and write each split to a JSON file.

    Parameters:
    - complete_datasets (dict): The dataset organized by datasets and splits.

    Return:
        None
    """
    for dataset in complete_datasets.values():
        for dataset_name, split in dataset.items():
            dataset_dict = {}  #create a new dictionary for the current dataset
            
            for split_name, memes in split.items():
                dataset_dict[dataset_name] = {} #create a new dictionary for the current split

                if split_name not in dataset_dict[dataset_name]:
                    dataset_dict[split_name] = {}

                for meme_info in memes.values():
                
                    meme_id = meme_info["meme id"]
                    meme_text = meme_info["meme text"].strip()
                    meme_caption = meme_info["meme caption"]
                    bert_representation = meme_info["text + caption (bert)"]
                    svm_representation = meme_info["text + caption (svm)"]

                    #add all meme info to the split dictionary
                    dataset_dict[split_name][meme_id] = {
                        "meme id": meme_id,
                        "meme text": meme_text,
                        "meme caption": meme_caption,
                        "bert representation": bert_representation,
                        "svm representation": svm_representation
                    }

                #save the split dictionary to a JSON file
                output_file = f"datasets/{dataset_name}_{split_name}.json"
                with open(output_file, "w", encoding='utf-8') as f:
                    json.dump(dataset_dict[split_name], f, ensure_ascii=False, indent=4)
                print(f"Saved {dataset_name} {split_name} split to {output_file}")

#add labels to datasets:

def add_labels(dataset_name, split_name):
    """
    Update the dataset with corresponding labels for EXIST2024 or MAMI.

    Return None. Update the dataset file in place and save the modified version.


    Parameters:
    - dataset_name (str): The name of the dataset. Possible values: "EXIST2024" or "MAMI".
    - split_name (str): The data split to process. Possible values: "training" or "test".
    """

    #load dataset json
    ds = f"datasets/{dataset_name}_{split_name}.json"

    with open(ds, "r", encoding="utf8") as fl:
        content = json.load(fl)

    if dataset_name == "EXIST2024":
        #set up categories in EXIST dataset
        categories = ["IDEOLOGICAL-INEQUALITY",
                        "MISOGYNY-NON-SEXUAL-VIOLENCE", 
                        "OBJECTIFICATION", 
                        "SEXUAL-VIOLENCE",
                        "STEREOTYPING-DOMINANCE"
                        ]

        if split_name == "training":

            labels_sexism_path = "../datasets/EXIST2021-2024_datasets/2024 EXIST/evaluation/golds/EXIST2024_training_task4_gold_hard.json"
            labels_type_path = "../datasets/EXIST2021-2024_datasets/2024 EXIST/evaluation/golds/EXIST2024_training_task6_gold_hard.json"

            with open(labels_sexism_path, "r", encoding="utf8") as fl:
                labels = json.load(fl)

            with open(labels_type_path, "r", encoding="utf8") as fl:
                labels_type = json.load(fl)

        #convert labels into dictionaries to update json
        #get sexist label per meme
        labels_dict = {item["id"]: "1" if "YES" in item["value"] else "0" for item in labels}
        #get category labels per meme
        labels_type_dict = {
            item["id"]: {cat.lower(): "1" if cat in item["value"] else "0" for cat in categories}
            for item in labels_type
        }

        # Update content with new labels
        for meme_id, meme_info in content.items():
            if meme_id in labels_dict:
                meme_info["sexist"] = labels_dict[meme_id]
            if meme_id in labels_type_dict:
                meme_info.update(labels_type_dict[meme_id])

    else: #MAMI
        categories = ["misogynous", "shaming", "stereotype", "objectification", "violence"]
        labels_dict = {} #create new dict to store the labels and update each meme

        # Determine file path based on split
        labels_path = (
            "../datasets/MAMI DATASET/TRAINING/training.csv"
            if split_name == "training"
            else "../datasets/MAMI DATASET/test_labels.txt"
        )

        with open(labels_path,"r",encoding="utf-8-sig") as file:
            lines = [line for line in file.read().split('\n') if line.strip()]  #exclude empty lines

        #test txt file does not have header, the rest is the same as training file
        for row in lines[1:] if split_name == "training" else lines:
            columns = row.strip("\n").split("\t")
            meme_id = columns[0] #get meme ID
            labels = [label for label in columns[1:6]] #get labels per meme
            labels_dict[meme_id] = dict(zip(categories, labels))

        for meme_id_, meme_info in content.items():
            if meme_id_ in labels_dict:
                #update each meme dict with the labels
                meme_info.update(labels_dict[meme_id_])
            

    #save the split dictionary to a JSON file
    with open(ds, "w", encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)
    print(f"Saved {dataset_name} {split_name} split to {ds}")


def add_meme_path(file_with_path,df_path):
    """
    Add meme image paths as a new column to a json-formatted DataFrame based on matching meme IDs.
    Save back the updated DataFrame to the original .json file.
    
    Return None. Print confirmation of saved json file with image paths.

    Parameters:
    ----------
    - file_with_path (str): Path to the json file containing meme metadata with 'meme id' and 'meme path'.
    - df_path (str): Path to the DataFrame json file to be updated with the meme paths.
    """

    df = pd.read_json(df_path,orient='index') #read df

    #create a dictionary with meme id: meme path
    meme_id_to_path = {}

    with open(file_with_path, "r", encoding="utf8") as fl:
            content = json.load(fl)

    for dataset in content.values():
        for split in dataset.values():
            for memes in split.values():
                for meme_info in memes.values():
                    meme_id = meme_info["meme id"]
                    meme_path = meme_info["meme path"]
                    meme_id_to_path[meme_id] = meme_path
    
    #get the meme paths by mapping the ids in the dataset df
    image_path = df["meme id"].astype(str).map(meme_id_to_path)
    #add the meme paths to df as new column (loc 1 after meme id)
    if "meme path" in df.columns:
        df.drop(columns=["meme path"], inplace=True)
    df.insert(loc=1, column="meme path", value=image_path)

    #return back to dict to save to json format
    df_dict = df.astype(str).to_dict(orient="index") 

    #save the split dictionary to a JSON file
    with open(df_path, "w", encoding='utf-8') as f:
        json.dump(df_dict, f, ensure_ascii=False, indent=4)
    print(f"Saved {df_path} with image paths")



### gold labels:

def load_dataset(dataset, data_dir="datasets",split_name = "training"):
    """
    Load a dataset from a JSON file and convert it into a pandas DataFrame.
    Return a pandas DataFrame containing the dataset loaded from the JSON file.

    Parameters:
    - dataset (str): The name of the dataset.
    - data_dir (str): The name of the directory where the datasets are located. Default to "datasets".
    - split_name (str): The specific split of the dataset. Default to "training".
    """

    #load dataset
    dataset_path = f"{data_dir}/{dataset}_{split_name}.json"
    with open(dataset_path, "r", encoding="utf8") as fl:
        content = json.load(fl)
    #convert to df
    df = pd.DataFrame.from_dict(content, orient="index")

    return df


def get_pyevall_evaluation_file(df,bin_label,labels,test_case,type_eval):
    """
    Convert a DataFrame of labeled meme dataset into a format suitable for PyEvALLEvaluation.

    Parameters:
    - df (pandas.DataFrame): A DataFrame containing meme id and associated labels. 
    - bin_label (str): The name of the binary label column in the DataFrame (sexist or misogynyous). 
                        This will be used to create binary categories ("yes" or "no") in the output 
                        if `type_eval` is set to `"binary"` or `"flat"`.
    - labels (list): A list of column names representing the labels for evaluation in the dataset.
    - test_case (str): The test case identifier to be added as a new column in the output DataFrame, e.g. "MAMI" or "EXIST2024".
    - type_eval (str): The type of evaluation format to be used. It can be:
        - `"binary"`: For binary classification.
        - `"hierarchical"`: For hierarchical multi-label classification.
         - `"flat"`: For a flat multi-label classification format, including binary classification in the labels.
    
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
    gold_labels = df[["meme id"]+labels]
    gold_labels.insert(0, "test_case", [test_case] * (len(gold_labels)), True) #add the test_case column as per the library requirements

    if type_eval == "binary":
        #binary labels
        bin_gold_labels = gold_labels[["test_case","meme id",bin_label]] #keep the binary label only
        bin_gold_labels = bin_gold_labels.replace({f"{bin_label}":"0"}, "no").replace({f"{bin_label}":"1"}, "yes") #convert values to yes and no
        bin_gold_labels.rename(columns={"meme id": "id",f"{bin_label}": "value"}, inplace=True) #rename the columns as the requirements
        gold_labels_df = bin_gold_labels

    if type_eval == "hierarchical":
        #muli-label categories
        filtered_categories = [item for item in labels if item != bin_label]  #remove misogynous/sexist category
        h_multilabel_all_gold_labels = gold_labels[["test_case","meme id"]+filtered_categories] #keep only the categories for the hierarchical multilabel classification
        value_cols = h_multilabel_all_gold_labels.columns[2:] #filter only category columns
        h_multilabel_all_gold_labels["value"] = h_multilabel_all_gold_labels[value_cols].apply(lambda row: list(row.index[row == "1"]), axis=1) #get list of labels per instance when the value is 1
        h_multilabel_all_gold_labels["value"] = h_multilabel_all_gold_labels["value"].apply(lambda x: ["no"] if x == [] else x) #convert empty labels (empty lists) to "no" category
        h_multilabel_all_gold_labels = h_multilabel_all_gold_labels[["test_case","meme id","value"]]
        h_multilabel_all_gold_labels.rename(columns={"meme id": "id"}, inplace=True) #rename the id column as the requirements
        gold_labels_df = h_multilabel_all_gold_labels

    if type_eval == "flat":
        # multi-label flat (all labels)
        flat_multilabel_gold_labels = gold_labels[["test_case","meme id"]+labels] #keep all categories
        value_cols = flat_multilabel_gold_labels.columns[2:] #filter only category columns
        flat_multilabel_gold_labels["value"] = flat_multilabel_gold_labels[value_cols].apply(lambda row: [
            "yes" if col == bin_label and val == "1"
            else "no" if col == bin_label and val == "0" 
            else col #keep category column names where label is 1
            for col, val in row.items() if val == "1"  or col == bin_label 
            ], axis=1) #get list of labels per instance when the value is 1. yes and no for binary label
        flat_multilabel_gold_labels = flat_multilabel_gold_labels[["test_case","meme id","value"]]
        flat_multilabel_gold_labels.rename(columns={"meme id": "id"}, inplace=True) #rename the id column as the requirements
        gold_labels_df = flat_multilabel_gold_labels
    
    gold_labels_list = gold_labels_df.to_dict(orient="records") #convert df into a list of dictionaries as per requirements
    
    return gold_labels_list


def get_txt_evaluation_file_pred(df,labels):
    """
    Convert a DataFrame of labeled meme dataset into a format suitable for MAMI evaluation.

    Parameters:
    - df (pandas.DataFrame): A DataFrame containing meme id and associated labels.
    - labels (list): A list of column names representing the labels for evaluation in the dataset.

    Return a df formatted according to the MAMI evaluation metrics including the meme ID and the labels.
    """

    #convert files to input required by mami evaluation
    labels_df = df[["meme id"]+labels].copy()
    labels_df["meme id"] = labels_df["meme id"].astype(str)  #convert "id" column to string values
    
    return labels_df


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
