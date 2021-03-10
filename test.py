import a2
from sklearn.svm import LinearSVC

gmbfile = open('/scratch/lt2222-v21-resources/GMB_dataset.txt', "r")
inputdata = a2.preprocess(gmbfile)
gmbfile.close()
print(inputdata[0:20])
print(inputdata.loc[inputdata["Tag"] != "O"])
#instances = a2.create_instances(inputdata)
#print(instances[0:20])

