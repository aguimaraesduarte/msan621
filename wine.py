# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics.cluster import v_measure_score
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

wine = pd.read_csv("wine.csv", header=None, names=["Type","1","2","3","4","5","6","7","8","9","10","11","12","13"])

X = wine[wine.columns[1:]]
Y = wine["Type"]

best_k = 0
best_v_measure = 0
for k in range(1,10):
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(X)
	# Each observation now “belongs” to a cluster. Which one?
	pred = kmeans.labels_
	# kmeans.predict(X) would give the same vector
	v_measure = v_measure_score(pred, Y)
	if v_measure > best_v_measure:
		best_v_measure = v_measure
		best_k = k
	print "k = %d\tV-measure = %.3f" %(k,v_measure)
print "Best k = %d" %best_k

kmeans = KMeans(n_clusters=best_k)
kmeans.fit(X)
# Each observation now “belongs” to a cluster. Which one?
pred = kmeans.labels_
# kmeans.predict(X) would give the same vector
v_measure = v_measure_score(pred, Y)


# Normalise the data
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


# Perform PCA
n = len(X.columns)
pca = PCA(n_components=n)
pca.fit(X_scaled)

xvector = pca.components_[0]
yvector = pca.components_[1]

xs = pca.transform(X_scaled)[:,0]
ys = pca.transform(X_scaled)[:,1]


# Draw the biplot
#for i in range(len(xvector)):
#	plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys), color='r', width=0.0005, head_width=0.0025)
#	plt.text(xvector[i]*max(xs), yvector[i]*max(ys), list(X.columns.values)[i], color='r')

for i in range(len(xs)):
	if Y[i] == 1:
		c = "b"
	elif Y[i] == 2:
		c = "r"
	else:
		c = "g"
	plt.plot(xs[i], ys[i], c)
	plt.text(xs[i], ys[i], pred[i], color=c)

plt.show()