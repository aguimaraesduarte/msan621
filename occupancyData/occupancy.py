import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

train = pd.read_table("datatraining.txt", sep=",")
test = pd.read_table("datatest.txt", sep=",")

X = train.loc[:,train.columns[1:-1]]
Y = train.loc[:,train.columns[-1]]
X_test = test.loc[:,test.columns[1:-1]]
Y_test = test.loc[:,test.columns[-1]]

# Decision Tree Classifier
print "Decision Tree Classifier"
best_depth = 1
best_acc = 0
for depth in range(1,11):
	print "Depth:", depth
	classifier = DecisionTreeClassifier(criterion="gini")
	classifier.fit(X, Y)
	cv = cross_val_score(classifier, X, Y, cv=4,
	                     scoring=metrics.make_scorer(metrics.accuracy_score))
	if cv.mean() > best_acc:
		best_acc = cv.mean()
		best_depth = depth
	print "Cross-val: ", cv.mean()
print "Best depth:", best_depth
print

# Print Best Tree
from StringIO import StringIO
import pydotplus 
from sklearn import tree

classifier = DecisionTreeClassifier(max_depth=best_depth)
classifier.fit(X, Y)
dotfile = StringIO()
tree.export_graphviz(classifier, out_file=dotfile)
graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
graph.write_pdf("tree.pdf")

# LDA
print "LDA"
classifier = LinearDiscriminantAnalysis()
classifier.fit(X, Y)
cv = cross_val_score(classifier, X, Y, cv=4,
                     scoring=metrics.make_scorer(metrics.accuracy_score))
print "Cross-val: ", cv.mean()
print

# Logit
print "Logit"
classifier = LogisticRegression()
classifier.fit(X, Y)
cv = cross_val_score(classifier, X, Y, cv=4,
                     scoring=metrics.make_scorer(metrics.accuracy_score))
print "Cross-val: ", cv.mean()
print

# Prediction in Test
print "Compare Models in Prediction"
for classifier in [DecisionTreeClassifier(),
                  LinearDiscriminantAnalysis(),
                  LogisticRegression()]:
	print str(classifier)[:str(classifier).find('(')]
	classifier.fit(X, Y)
	pred = classifier.predict(X_test)
	acc = metrics.accuracy_score(Y_test, pred)
	print "Accuracy:", acc
