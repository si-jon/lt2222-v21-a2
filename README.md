# LT2222 V21 Assignment 2

Assignment 2 from the course LT2222 in the University of Gothenburg's winter 2021 semester.

Your name: Sigrid Jonsson


## Preprocessing

In the preprocessing step, the words are lemmatized and lowercased. I decided to remove punctuation; since the sentence numbers are included in the data, the punctuation isn't needed to be able to differentiate the sentences. It also makes the creation of instances in the next step easier.
I also decided to remove the non first words of the named entities, that is the I-tags. These are not to be included as features, and the B-tags already contains the information of the type, the I-tags are made redundant. To remove these also makes the next step easier, since the case of several NEs in a row no longer needs to be taken in to account.


## NE class instances

* Include other named entities as features.


## Data table

* Pick the top frequent words to include in the table
* Reduces the table using sklearn


## Confusion matrix

* Number of NE types are not evenly distributed: more than half of the data is geo.
* Training and testing data seems to roughly correspond to each other; kind of the same distribution of true positives, true negatives, false positives, and false negatives.