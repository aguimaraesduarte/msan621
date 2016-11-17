import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import linear_model

names = ["v%d" %i for i in range(281)]
train = pd.read_table("blogData_train.csv", sep=",", header=None, names=names)
test = pd.read_table("blogData_test-2012.03.31.01_00.csv", sep=",", header=None, names=names)

train_X = train[names[:-1]]
train_Y = train[names[-1]]
test_X = test[names[:-1]]
test_Y = test[names[-1]]

depth=3
regressor = DecisionTreeRegressor(max_depth=depth)
regressor.fit(train_X, train_Y)
hypotheses = regressor.predict(test_X)
print "Depth: %d\tMSE: %.3f" % (depth, metrics.mean_squared_error(test_Y, hypotheses))

algo = linear_model.LinearRegression()
algo.fit(train_X, train_Y)
hypotheses = algo.predict(test_X)
print "MSE: %.3f" % metrics.mean_squared_error(test_Y, hypotheses)

from StringIO import StringIO
import pydotplus 
from sklearn import tree

dotfile = StringIO()
tree.export_graphviz(regressor, out_file=dotfile)
graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
graph.write_pdf("tree.pdf")