import sys
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd
import numpy as np
from sklearn import preprocessing


# Import the data
# Data is assumed to be in format: State (label) with features as real numbers, after a header row
my_csv = sys.argv[1]
df = pd.read_csv(my_csv, index_col=0)


# Normalise the data
scaler = preprocessing.StandardScaler()
data_only = np.array(df)
scaler.fit(data_only)
dat = scaler.transform(data_only)


# Perform PCA
n = len(df.columns)
pca = sklearnPCA(n_components=n)
pca.fit(dat)

xvector = pca.components_[0]
yvector = pca.components_[1]

xs = pca.transform(dat)[:,0]
ys = pca.transform(dat)[:,1]


# Draw the biplot
for i in range(len(xvector)):
	plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys), color='r', width=0.0005, head_width=0.0025)
	plt.text(xvector[i]*max(xs), yvector[i]*max(ys), list(df.columns.values)[i], color='r')

for i in range(len(xs)):
	plt.plot(xs[i], ys[i], 'bo')
	plt.text(xs[i], ys[i], list(df.index)[i], color='b')

plt.show()