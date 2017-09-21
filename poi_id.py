#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
sys.path.append("../tools/")



from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi", "ratio_from_poi_email", "ratio_to_poi_email",
                 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### look at data
#print len(data_dict.keys())
#print data_dict['TOTALS', 'YEAP SOON', 'THE TRAVEL AGENCY IN THE PARK', 'BUY RICHARD B']
#print data_dict.values()

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

features = ["salary", "bonus"]
data_dict.pop('TOTAL',0)
data_dict.pop('YEAP SOON', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data = featureFormat(data_dict, features)

### remove NAN's from dataset
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
### plot features after removing outliers
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
#plt.show()

###create new features 'ratio_from_poi_email' and 'ratio_to_poi_email'
def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

###create lists of new features
ratio_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
ratio_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["ratio_from_poi_email"]=ratio_from_poi_email[count]
    data_dict[i]["ratio_to_poi_email"]=ratio_to_poi_email[count]
    count +=1

my_dataset = data_dict

### and extract features specified from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)

### plot new features
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter( from_poi, to_poi )
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("ratio of emails this person gets from poi")
plt.ylabel("ration of emails this person sends to poi")
#plt show()



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
features_list = ["poi", "salary", "bonus", "ratio_from_poi_email", "ratio_to_poi_email",
                'deferral_payments', 'total_payments', 'total_stock_value', 'shared_receipt_with_poi', 'restricted_stock', 
                 'exercised_stock_options', 'restricted_stock_deferred', 'expenses', 'loan_advances',
                 'other', 'director_fees', 'deferred_income', 'long_term_incentive']
my_dataset = data_dict
data = featureFormat(my_dataset, features_list)

### split into labels and features 
labels, features = targetFeatureSplit(data)

### split data into training and testing datasets
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier

t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
pred= clf.predict(features_test)
print 'accuracy', score
print 'precision = ', precision_score(labels_test,pred)
print 'recall = ', recall_score(labels_test,pred)
print "Decision tree algorithm time:", round(time()-t0, 3), "s"



importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(16):
    print "{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]])




### try Naive Bayes for prediction
t0 = time()

clf= GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
print accuracy

print "NB algorithm time:", round(time()-t0, 3), "s"


from sklearn import metrics
#print "precision:", metrics.precision_score(labels_test, predictions)
#print "recall:", metrics.recall_score(labels_test, predictions)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### split data into training and testing datasets
features_list = ["poi", "ratio_from_poi_email", "ratio_to_poi_email", 'shared_receipt_with_poi']

my_dataset = data_dict
data = featureFormat(my_dataset, features_list)

### split into labels and features
labels, features = targetFeatureSplit(data)

### deploying feature selection
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

### use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

from sklearn.tree import DecisionTreeClassifier

t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
print 'accuracy before tuning ', score

print "Decision tree algorithm time:", round(time()-t0, 3), "s"

### use manual tuning parameter min_samples_split
t0 = time()
clf = DecisionTreeClassifier(min_samples_split=8)
clf = clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
print("done in %0.3fs" % (time() - t0))

acc=accuracy_score(labels_test, pred)

print "Validating algorithm:"
print "accuracy after tuning = ", acc
print 'precision = ', precision_score(labels_test,pred)
print 'recall = ', recall_score(labels_test,pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)