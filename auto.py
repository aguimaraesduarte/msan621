import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale # For normalizing

names=["mpg","cylinders","displacement","horsepower","weight","acceleration","year","origin","name"]
auto = pd.read_table("auto.csv", sep=",", header=None, names=names)

auto = scale(auto.values[:, 1:-1])

pca = PCA(n_components=components)
pca.fit(auto)
variance = pca.explained_variance_ratio_

cumvariances = np.cumsum(variance)

plt.plot(cumvariances)
plt.show()