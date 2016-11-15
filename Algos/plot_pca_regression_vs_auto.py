def read_csv (file_name):
	import csv

	with open(file_name) as csvfile:
		X = []
		Y = []
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			Y.append(float(row[0]))
			X.append(map(float, row[1:-1]))
	return X, Y


def plot_pca_vals(X):
	from sklearn.decomposition import PCA
	from sklearn.preprocessing import scale
	import matplotlib.pyplot as plt

	pca = PCA(n_components=7)
	pca.fit(X)

	var = pca.explained_variance_ratio_
	print var
	var_sum = []
	cumulative = 0.0
	for val in var:
		var_sum.append(val + cumulative)
		cumulative += val
	plt.plot(var_sum)
	plt.show()



X, Y = read_csv("auto.csv")
plot_pca_vals(X)