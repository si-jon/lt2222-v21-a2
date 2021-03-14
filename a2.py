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
        str: Lemmatized word, if it can be lemmatized. Otherwise the
        word unchanged.
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
    """Return preproccessed data from inputfile as a table
    
    The preprocessing steps include:
        * Split the data from input to lists
        * Tokenize the data of each row
        * Lemmatize the words
        * Remove punctuations
        * Remove I-xyz; since they are not to be included as feature
        they aren't necessary when creating instances

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
    """Create instances of NE classes with corresponding list of features

    The instance creation is done in the following steps:
        * Get all NE rows from data
        * For each NE in NE rows
            * Get all rows that are in the same sentence as NE
            * Of those rows, remove potential foregin NEs
            * From the remaining rows, get the rows surrounding NE,
            five before/after
            * Create list of words from the rows as features
            * If there is less than five word features of before/after, 
            sentence start/end, add corresponding start/end tags
            * Create instance of NE class and feature list

    Args:
        data (pandas.DataFrame): Table of data, where each row contains
                                information of each word and potential NEs 

    Returns:
        list: A list of instances of NE classes and features
    """
    start_tags = ['<S1>', '<S2>', '<S3>', '<S4>', '<S5>']
    end_tags = ['<E5>', '<E4>', '<E3>', '<E2>', '<E1>']
    feat_count = 5
    instances = []
    # Get all rows containing NEs
    ne_rows = data.loc[data['Tag'].str.startswith('B')]
    for i, row in ne_rows.iterrows():
        # Get all rows included in the sentence
        sentence = data.loc[data['Sentence #'] == row['Sentence #']]

        # Remove all the other NEs from the sentence
        sentence_nes = sentence.loc[(sentence['Tag'].str.startswith('B')) & (
            sentence.index != row.name)]
        sentence_culled = sentence.drop(sentence_nes.index, axis='index')

        # Get the surrounding feature rows, five before and after the NE
        ne_pos = sentence_culled.index.get_loc(row.name)
        start = max(ne_pos-feat_count, 0)
        end = ne_pos+1+feat_count
        feat_rows = sentence_culled.iloc[start:end]

        # Create list of features out of feature rows
        ne_pos = feat_rows.index.get_loc(row.name)
        row_count = len(feat_rows.index)
        feat_list = feat_rows['Word'].tolist()
        feat_list = feat_list[:ne_pos] + feat_list[(ne_pos + 1):]
        
        # If there are features missing (end/start of sentence), 
        # prepend and append start/end tags
        start_tag_count = ne_pos
        end_tag_count = feat_count + ne_pos + 1 - row_count
        features = start_tags[start_tag_count:] + \
            feat_list + end_tags[:end_tag_count]
            
        # Create instance
        neclass = row['Tag'][2:]
        instances.append(Instance(neclass, features))

    return instances


def get_topfreq_feats(df, top_freq=3000):
    """Get dataframe containing the top frequent features

    Sorts the columns of the incoming dataframe df by the sum of the values
    of each column. From this sorted dataframe, return slice it to keep
    the top frequent columns.

    Args:
        df (pandas.DataFrame): Table to get the top frequent feats from
        top_freq (int, optional): The amount of columns to return. 
                                  Defaults to 3000.

    Returns:
        pandas.DataFrame: Table only including the top frequent columns.
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
    """Dimensionality reduction of table

    Args:
        df (pandas.DataFrame): Table to reduce
        dims (int, optional): Dimension of the reduced table. Defaults to 300.

    Returns:
        pandas.DataFrame: A reduced table
    """
    df_copy = df.drop(columns=CLASS, inplace=False)
    svd = TruncatedSVD(n_components=dims)
    reduced = svd.fit_transform(df_copy)
    df_reduced = pd.DataFrame(reduced)
    df_reduced.insert(loc=0, column=CLASS, value=df[CLASS])
    return df_reduced


def create_table(instances):
    """Creates table instances

    Creates humongous table out of NE classes and features from instances.
    The table columns are the different features, and each row contains
    the number of each feature the corresponding NE class have.

    Args:
        instances (list): List of instances, generated from create_instance

    Returns:
        pandas.DataFrame: A reduced table of the ne classes and their features
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
        bigdf (pandas.DataFrame): Table to be split

    Returns:
        Tuple: A tuple of four containing a table of training data, a list
        of NE classes corresponding to the training data, a table of testing data,
        and a list of NE classes corresponding to the testing data
    """
    df_train = bigdf.sample(frac=0.80)
    df_test = bigdf.drop(df_train.index)
    return df_train.drop(CLASS, axis=1).to_numpy(), df_train[CLASS], df_test.drop(CLASS, axis=1).to_numpy(), df_test[CLASS]


def confusion_matrix(truth, predictions):
    """Creates a confusion matrix

    Args:
        truth (list): The actual NE classes
        predictions (list): The predicted NE classes

    Returns:
        pandas.DataFrame: A confusion matrix
    """
    neclasses = list(set(truth.tolist() + predictions.tolist()))
    con_matrix = sklearn_confusion_matrix(truth, predictions, labels=neclasses)
    df = pd.DataFrame(data=con_matrix, index=neclasses, columns=neclasses)
    return df


def bonusb(filename):
    pass
