import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn.cross_validation import cross_val_score

names = ["v%d" %i for i in range(281)]
train = pd.read_table("blogData_train.csv", sep=",", header=None, names=names)
test = pd.read_table("blogData_test-2012.03.31.01_00.csv", sep=",", header=None, names=names)
"""
train_X = train[names[:-1]]
train_Y = train[names[-1]]
test_X = test[names[:-1]]
test_Y = test[names[-1]]

depth=3
regressor = DecisionTreeRegressor(max_depth=depth)
regressor.fit(train_X, train_Y)
hypotheses = regressor.predict(test_X)
print "Depth: %d\tMSE: %.3f" % (depth, metrics.mean_squared_error(test_Y, hypotheses))

from StringIO import StringIO
import pydotplus 
from sklearn import tree

dotfile = StringIO()
tree.export_graphviz(regressor, out_file=dotfile)
graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
graph.write_pdf("tree.pdf")
"""
"""
algo = linear_model.LinearRegression()
algo.fit(train_X, train_Y)
hypotheses = algo.predict(test_X)
print "MSE: %.3f" % metrics.mean_squared_error(test_Y, hypotheses)
"""

X = train.loc[:,train.columns[:-1]]
Y = train.loc[:,train.columns[-1]]
X_test = test.loc[:,test.columns[:-1]]
Y_test = test.loc[:,test.columns[-1]]

# Decision Tree Regressor
print "Decision Tree Regressor"
best_depth = 1
best_MSE = 1e50
for depth in range(1,11):
	print "Depth:", depth
	regressor = DecisionTreeRegressor(max_depth=depth)
	regressor.fit(X, Y)
	cv = cross_val_score(regressor, X, Y, cv=4,
	                     scoring=metrics.make_scorer(metrics.mean_squared_error))
	if cv.mean() < best_MSE:
		best_MSE = cv.mean()
		best_depth = depth
	print "Cross-val: ", cv.mean()
print "Best depth:", best_depth
print

# Print Best Tree
from StringIO import StringIO
import pydotplus 
from sklearn import tree

regressor = DecisionTreeRegressor(max_depth=best_depth)
regressor.fit(X, Y)
dotfile = StringIO()
tree.export_graphviz(regressor, out_file=dotfile)
graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
graph.write_pdf("tree.pdf")

# Ridge Regressor
print "Ridge Regressor"
best_ridge_alpha = 1
best_MSE = 1e50
for alpha in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]:
	print "Alpha:", alpha
	regressor = linear_model.Ridge(alpha = alpha)
	regressor.fit(X, Y)
	cv = cross_val_score(regressor, X, Y, cv=4,
	                     scoring=metrics.make_scorer(metrics.mean_squared_error))
	if cv.mean() < best_MSE:
		best_MSE = cv.mean()
		best_ridge_alpha = alpha
	print "Cross-val: ", cv.mean()
print "Best Ridge Alpha:", best_ridge_alpha
print

# Lasso Regressor
print "Lasso Regressor"
best_lasso_alpha = 1
best_MSE = 1e50
for alpha in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]:
	print "Alpha:", alpha
	regressor = linear_model.Lasso(alpha = alpha)
	regressor.fit(X, Y)
	cv = cross_val_score(regressor, X, Y, cv=4,
	                     scoring=metrics.make_scorer(metrics.mean_squared_error))
	if cv.mean() < best_MSE:
		best_MSE = cv.mean()
		best_lasso_alpha = alpha
	print "Cross-val: ", cv.mean()
print "Best Lasso Alpha:", best_lasso_alpha
print

# Prediction in Test
print "Compare Models in Prediction"
for regressor in [linear_model.LinearRegression(),
                  linear_model.Ridge(alpha = best_ridge_alpha),
                  linear_model.Lasso(best_lasso_alpha),
                  DecisionTreeRegressor(max_depth=best_depth)]:
	print str(regressor)[:str(regressor).find('(')]
	regressor.fit(X, Y)
	pred = regressor.predict(X_test)
	mse = metrics.mean_squared_error(Y_test, pred)
	print "MSE:", mse