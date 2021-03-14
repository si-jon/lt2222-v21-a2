# LT2222 V21 Assignment 2

Assignment 2 from the course LT2222 in the University of Gothenburg's winter 2021 semester.

By Sigrid Jonsson


## Preprocessing

In the preprocessing step, the words are lemmatized and lowercased. I decided to remove punctuation; since the sentence numbers are included in the data, the punctuation isn't needed to be able to differentiate the sentences. It also makes the creation of instances in the next step easier.
I also decided to remove the non first words of the named entities, that is the I-tags. These are not to be included as features, and the B-tags already contains the information of the type, the I-tags are made redundant. To remove these also makes the next step easier, since the case of several NEs in a row no longer needs to be taken in to account.


## Features

Features I included for each NE was the ten words surrounding the NE, five words on either side, that was part of the same sentence. If the NE was in the beginning or end of a sentence, I inserted corresponding start or end tags. I decided to exclude other NEs as part of the features, since they are infrequent and the same NE will likely not appear twice. 


## Confusion matrix

The first think I notice in the confusion matrix is how unevenly the number of NE types are in the data. More than half of the data is of the type "geo". 
The predictions on the training and testing data are almost equally bad. The NE types there are very few of in the data, eg "eve", "nat", and "art", the predictions on the training data are more correct than on the testing data.