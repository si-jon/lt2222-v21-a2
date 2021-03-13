import a2
from sklearn.svm import LinearSVC

gmbfile = open('/scratch/lt2222-v21-resources/GMB_dataset.txt', "r")
inputdata = a2.preprocess(gmbfile)
gmbfile.close()
instances = a2.create_instances(inputdata)
bigdf = a2.create_table(instances)
train_X, train_y, test_X, test_y = a2.ttsplit(bigdf)
model = LinearSVC()
model.fit(train_X, train_y)
train_predictions = model.predict(train_X)
test_predictions = model.predict(test_X)

print(a2.confusion_matrix(test_y, test_predictions))
print(a2.confusion_matrix(train_y, train_predictions))