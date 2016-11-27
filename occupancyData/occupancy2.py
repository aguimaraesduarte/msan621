import pandas as pd
import numpy as np
import pylab as pl
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, metrics
from sklearn.cross_validation import KFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score

train = pd.read_csv('datatraining.txt', header=0)
test = pd.read_csv('datatest.txt', header=0)

Y_train = train.iloc[:, -1]
Y_test = test.iloc[:, -1]
X_train = train.iloc[:, 1:-1]
X_test = test.iloc[:, 1:-1]

#X_train['date'] = pd.to_datetime(X_train['date'])
#X_test['date'] = pd.to_datetime(X_train['date'])

#X_train.dtypes

# Decision Tree Classifier
print
print "Find Best Depth through CV"
best_depth = 1
best_acc = 0
for depth in range(1,11):
  print "Depth:", depth
  classifier = DecisionTreeClassifier(criterion="gini")
  classifier.fit(X_train, Y_train)
  cv = cross_val_score(classifier, X_train, Y_train, cv=4,
                       scoring=metrics.make_scorer(metrics.accuracy_score))
  if cv.mean() > best_acc:
    best_acc = cv.mean()
    best_depth = depth
  print "Cross-val: ", cv.mean()
print "Best depth:", best_depth

print
print "Decision Tree Classifier"
crit = "gini"
regressor = DecisionTreeClassifier(max_depth=best_depth, criterion=crit)
regressor.fit(X_train, Y_train)
acc = regressor.score(X_test, Y_test)
print "Accuracy:", acc

# LDA
print
print "LDA"
classifier = LinearDiscriminantAnalysis()
classifier.fit(X_train, Y_train)
cv = cross_val_score(classifier, X_train, Y_train, cv=4,
                     scoring=metrics.make_scorer(metrics.accuracy_score))
print "Cross-val: ", cv.mean()

# Logit
print
print "Logit"
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
cv = cross_val_score(classifier, X_train, Y_train, cv=4,
                     scoring=metrics.make_scorer(metrics.accuracy_score))
print "Cross-val: ", cv.mean()

# Bagging
print
print "Bagging"
estimators = 10
classifier = BaggingClassifier(n_estimators=estimators)
classifier.fit(X_train, Y_train)
acc = classifier.score(X_test, Y_test)
print "Accuracy:", acc

# Random forests
print
print "Random Forests"
estimators = 10
classifier = RandomForestClassifier(n_estimators=estimators)
classifier.fit(X_train, Y_train)
acc = classifier.score(X_test, Y_test)
print "Accuracy:", acc

print
print "Optimal Learning Rate"
print "Rough estimate"
tuned_parameters = {'learning_rate':np.arange(0.1,1.1,0.1)}
clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters)
clf.fit(X_train, Y_train)
print clf.best_params_
#for score in clf.grid_scores_:
#    print score

print "Fine tuning"
tuned_parameters = {'learning_rate':np.arange(0.01,0.11,0.01)}
clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters)
clf.fit(X_train, Y_train)
print clf.best_params_
#for score in clf.grid_scores_:
#    print score

# AdaBoost
print
print "AdaBoost"
rate = clf.best_params_['learning_rate']
classifier = AdaBoostClassifier(learning_rate=rate)
classifier.fit(X_train, Y_train)
acc = classifier.score(X_test, Y_test)
print "Accuracy:", acc

print 
print "Compare models"
for classifier in [DecisionTreeClassifier(),
                  LogisticRegression(),
                  LinearDiscriminantAnalysis(), 
                  QuadraticDiscriminantAnalysis(), 
                  KNeighborsClassifier(),
                  AdaBoostClassifier(),
                  BaggingClassifier(),
                  RandomForestClassifier(),
                  ]:
  print str(classifier)[:str(classifier).find('(')],
  classifier.fit(X_train, Y_train)
  pred = classifier.predict(X_test)
  acc = metrics.accuracy_score(Y_test, pred)
  print "Accuracy:", acc