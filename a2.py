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

lemmatizer = WordNetLemmatizer()

# Module file with functions that you fill in so that they can be
# called by the notebook.  This file should be in the same
# directory/folder as the notebook when you test with the notebook.

# You can modify anything here, add "helper" functions, more imports,
# and so on, as long as the notebook runs and produces a meaningful
# result (though not necessarily a good classifier).  Document your
# code with reasonable comments.

def get_lemma(word, tag):
    if tag.startswith('J'): 
        simple_tag =  wordnet.ADJ 
    elif tag.startswith('V'): 
        simple_tag =  wordnet.VERB 
    elif tag.startswith('N'): 
        simple_tag =  wordnet.NOUN 
    elif tag.startswith('R'): 
        simple_tag = wordnet.ADV 
    else:
        return word

    return lemmatizer.lemmatize(word, simple_tag)

# Function for Part 1
# 1. To preprocess the text (lowercase and lemmatize; punctuation can be preserved as it gets its own rows).
def preprocess(inputfile):
    header_text = inputfile.readline().strip()
    header_list = re.split("[\t]", header_text)
    data_text = inputfile.readlines()
    data_list = []
    for line in data_text:
        dl = line.split()
        if not dl[2].isalnum():
            continue
        lemma = get_lemma(dl[2], dl[3]).lower()
        dl[2] = lemma
        data_list.append(dl[1:])

    df = pd.DataFrame(data=data_list, columns=header_list)
    return df


def preprocess_old(inputfile):
    in_data = inputfile.readlines()
    out_data = []

    for line in in_data:
        data_list = line.split()
        lemma = get_lemma(data_list[2], data_list[3]).lower()
        data_list[2] = lemma
        out_data.append(data_list)
    return out_data

# Code for part 2
# 2. To create instances from every identified named entity in the text with the type of the NE as the class, and a surrounding context of five words on either side as the features.
class Instance:
    def __init__(self, neclass, features):
        self.neclass = neclass
        self.features = features

    def __str__(self):
        return "Class: {} Features: {}".format(self.neclass, self.features)

    def __repr__(self):
        return str(self)

def create_instances(data):
    start_tags = ["<S5>", "<S4>", "<S3>", "<S2>", "<S1>"]
    end_tags = ["<E5>", "<E4>", "<E3>", "<E2>", "<E1>"]
    instances = []
    #for idx, val in enumerate(data):
    #    if 
    return None

def create_instances_old(data):
    start_tags = ["<S5>", "<S4>", "<S3>", "<S2>", "<S1>"]
    end_tags = ["<E5>", "<E4>", "<E3>", "<E2>", "<E1>"]
    punct_list = [".", "?", "!", ";", ":"]
    instances = []
    for idx, val in enumerate(data):
        if val[4].startswith("I"):
            continue
        elif val[4].startswith("B"):
            start_id = idx - 1
            end_id = idx + 1
            for someotheridx in range(1,6):
                thisotheridx = idx + someotheridx
                if not data[thisotheridx][4].startswith("I"):
                    end_id = thisotheridx
                    break
            neclass = val[4][2:]
            pre_features = []
            post_features = []
            start_reached = False
            stop_reached = False
            s_tag_idx = 0
            e_tag_idx = 0
            for i in range(0,5):
                if start_reached:
                    pre_features.insert(0, start_tags[s_tag_idx])
                    s_tag_idx += 1
                else:
                    pre_feat = data[start_id-i]
                    if pre_feat[2] in punct_list:
                        start_reached = True
                        pre_features.insert(0, start_tags[s_tag_idx])
                        s_tag_idx += 1
                    else:
                        pre_features.insert(0, pre_feat[2])

                if stop_reached:
                    post_features.append(end_tags[e_tag_idx])
                    e_tag_idx += 1
                else:
                    post_feat = data[end_id+i]
                    if post_feat[2] in punct_list:
                        stop_reached = True
                        post_features.append(end_tags[e_tag_idx])
                        e_tag_idx += 1
                    else:
                        post_features.append(post_feat[2])

            features = pre_features + post_features
            instances.append(Instance(neclass, features))

    return instances

# Code for part 3
# 3. To generate vectors and split the instances into training and testing datasets at random.
def create_table(instances):
    df = pd.DataFrame()
    df['class'] = [random.choice(['art','eve','geo','gpe','nat','org','per','tim']) for i in range(100)]
    for i in range(3000):
        df[i] = npr.random(100)

    return df

def ttsplit(bigdf):
    df_train = pd.DataFrame()
    df_train['class'] = [random.choice(['art','eve','geo','gpe','nat','org','per','tim']) for i in range(80)]
    for i in range(3000):
        df_train[i] = npr.random(80)

    df_test = pd.DataFrame()
    df_test['class'] = [random.choice(['art','eve','geo','gpe','nat','org','per','tim']) for i in range(20)]
    for i in range(3000):
        df_test[i] = npr.random(20)
        
    return df_train.drop('class', axis=1).to_numpy(), df_train['class'], df_test.drop('class', axis=1).to_numpy(), df_test['class']

# Code for part 5
# 5. To evaluate the performance of the classifier.
def confusion_matrix(truth, predictions):
    print("I'm confusing.")
    return "I'm confused."

# Code for bonus part B
def bonusb(filename):
    pass
