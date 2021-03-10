# LT2222 V21 Assignment 2

Assignment 2 from the course LT2222 in the University of Gothenburg's winter 2021 semester.

Your name: Sigrid Jonsson

*Answer all questions in the notebook here.  You should also write whatever high-level documentation you feel you need to here.*


1. To preprocess the text (lowercase and lemmatize; punctuation can be preserved as it gets its own rows).

2. To create instances from every from every identified named entity in the text with the type of the NE as the class, and a surrounding context of five words on either side as the features.

3. To generate vectors and split the instances into training and testing datasets at random.

4. To train a support vector machine (via sklearn.svm.LinearSVC) for classifying the NERs.

5. To evaluate the performance of the classifier.