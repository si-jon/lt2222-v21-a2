import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

lemmatizer = WordNetLemmatizer()
CLASS = 'class_label'


def get_lemma(word, pos_tag):
    """Get the lemma of the word based on its POS tag

    Args:
        word (str): The word to lemmatize
        pos_tag (str): POS tag of the word to lemmatize

    Returns:
        str: Lemmatized word
    """
    if pos_tag.startswith('J'):
        simple_tag = wordnet.ADJ
    elif pos_tag.startswith('V'):
        simple_tag = wordnet.VERB
    elif pos_tag.startswith('N'):
        simple_tag = wordnet.NOUN
    elif pos_tag.startswith('R'):
        simple_tag = wordnet.ADV
    else:
        return word

    return lemmatizer.lemmatize(word, simple_tag)


def preprocess(inputfile):
    """Do some preprocessing of the data from inputfile

    * Split the data in the inputfile to list
    * lemmatize words
    * Removes punctuations
    * Removes I-xyz; since they are not to be included as feature, they aren't necessary when creating instances

    Args:
        inputfile (file): An open file handle

    Returns:
        pandas.DataFrame: The preprocessed data from inputfile
    """
    header_text = inputfile.readline().strip()
    header_list = re.split('[\t]', header_text)
    data_text = inputfile.readlines()
    data_list = []
    for line in data_text:
        dl = line.split()
        if not dl[2].isalnum() or dl[4].startswith('I'):
            continue
        lemma = get_lemma(dl[2], dl[3]).lower()
        dl[2] = lemma
        data_list.append(dl[1:])

    df = pd.DataFrame(data=data_list, columns=header_list)
    return df


class Instance:
    def __init__(self, neclass, features):
        self.neclass = neclass
        self.features = features

    def __str__(self):
        return "Class: {} Features: {}".format(self.neclass, self.features)

    def __repr__(self):
        return str(self)


def create_instances(data):
    """Get instances of NE classes with corresponding list of features

    * Get all NE rows from data
    * For each NE index in NE rows, get the 10 rows from data surronding the NE index
    * Remove the rows that is not in the same sentence as the current NE
    * Of the rows that are left, att the words to feature list
    * For the rows that was removed, add start tags and end to the feature list
    * Create instance of NE class and feature list


    Args:
        data (pandas.DataFrame): Assumes the format as is returned from preprocess 

    Returns:
        list: A list of instances
    """
    start_tags = ['<S1>', '<S2>', '<S3>', '<S4>', '<S5>']
    end_tags = ['<E5>', '<E4>', '<E3>', '<E2>', '<E1>']
    feat_count = 5
    instances = []
    ne_rows = data.loc[data['Tag'].str.startswith('B')]
    for i, row in ne_rows.iterrows():
        feat_rows = data.iloc[(i-feat_count):(i+1+feat_count)]
        feat_rows_culled = feat_rows.loc[feat_rows['Sentence #']
                                         == row['Sentence #']]
        row_count = len(feat_rows_culled.index)

        ne_pos = feat_rows_culled.index.get_loc(row.name)
        feat_list = feat_rows_culled['Word'].tolist()
        feat_list = feat_list[:ne_pos] + feat_list[(ne_pos + 1):]
        start_tag_count = ne_pos
        end_tag_count = feat_count + ne_pos + 1 - row_count

        features = start_tags[start_tag_count:] + \
            feat_list + end_tags[:end_tag_count]
        neclass = row['Tag'][2:]
        instances.append(Instance(neclass, features))

    return instances


def get_topfreq_feats(df, top_freq=3000):
    """Get dataframe containing the top frequent features

    * Calculate sum of each column
    * Create sorted dictionary of the columns 
    * Cast to list, and slice to only keep the top frequent
    * Copy the columns from df with the top frequent words to new dataframe

    Args:
        df (pandas.DataFrame): [description]
        top_freq (int, optional): [description]. Defaults to 3000.

    Returns:
        pandas.DataFrame: [description]
    """
    column_sums = {}
    for column in df.keys():
        if column is CLASS:
            continue
        column_sums[column] = df[column].sum()

    sorted_sums = dict(
        sorted(column_sums.items(), key=lambda item: item[1], reverse=True))
    top_freq_feats = list(sorted_sums)[:top_freq]
    top_freq_feats.insert(0, CLASS)
    df_topfreq_feats = df[top_freq_feats].copy()

    return df_topfreq_feats


def reduce(df, dims=300):
    """[summary]

    Args:
        matrix (pandas.DataFrame): [description]
        dims (int, optional): [description]. Defaults to 300.

    Returns:
        pandas.DataFrame: [description]
    """    
    df_copy = df.drop(columns=CLASS, inplace=False)
    svd = TruncatedSVD(n_components=dims)
    reduced = svd.fit_transform(df_copy)
    df_reduced = pd.DataFrame(reduced)
    df_reduced.insert(loc=0, column=CLASS, value=df[CLASS])
    return df_reduced

def create_table(instances):
    """Creates table instances

    * Makes list of instances to dict
    * Create dataframe out of dict
    * Gets the top frequent columns

    Args:
        instances (list): List of instances, generated from create_instance

    Returns:
        pandas.DataFrame: [description]
    """
    instance_data = []
    for inst in instances:
        feat_dict = {CLASS: inst.neclass}
        for feat in inst.features:
            if feat not in feat_dict:
                feat_dict[feat] = 0
            feat_dict[feat] += 1
        instance_data.append(feat_dict)

    df = pd.DataFrame.from_dict(data=instance_data).fillna(0)
    df_topfreq = get_topfreq_feats(df)
    df_reduced = reduce(df_topfreq)
    return df_reduced


def ttsplit(bigdf):
    """Splits a table into training and testing data

    Args:
        bigdf (pandas.DataFrame): [description]

    Returns:
        [type]: [description]
    """
    df_train = bigdf.sample(frac=0.80)
    df_test = bigdf.drop(df_train.index)
    return df_train.drop(CLASS, axis=1).to_numpy(), df_train[CLASS], df_test.drop(CLASS, axis=1).to_numpy(), df_test[CLASS]


def confusion_matrix(truth, predictions):
    """Creates a confusion matrix

    Args:
        truth (list): [description]
        predictions (list): [description]

    Returns:
        pandas.DataFrame: [description]
    """
    neclasses = list(set(truth.tolist() + predictions.tolist()))
    con_matrix = sklearn_confusion_matrix(truth, predictions, labels=neclasses)
    df = pd.DataFrame(data=con_matrix, index=neclasses, columns=neclasses)
    return df


def bonusb(filename):
    pass
