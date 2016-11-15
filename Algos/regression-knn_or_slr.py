

# Prints some "how to use this script" instructions
def help ():
	print 'Arguments by position:'
	print '\t1: Training file - Full or relative path'
	print '\t2: Test file     - Full or relative path'
	print '\t3: Algorithm     - One of {SLR|kNN}'


# Populates and returns a dictionary with options from command line
# Replace this with argparse (or getopt)
# cf. https://docs.python.org/2/howto/argparse.html
def get_args_by_position (args):
	opts = {}

	if len(args) != 4 and len(args) != 5:
		return opts
	opts['train_file'] = args[1]
	opts['test_file'] = args[2]

	if args[3].lower() == 'slr':
		opts['algo'] = 'SLR'
	elif args[3].lower() == 'knn':
		opts['algo'] = 'KNN'

	if len(args) == 5:
		opts['algo_arg'] = args[4]

	return opts


# Plot a (2D) graph of the data
# Uses matplotlib
def plot (data, targets):
	import matplotlib.pyplot as plt

	plt.scatter(data, targets, color='black')
	plt.xticks(())
	plt.yticks(())

	plt.show()


# Reads a CSV file
# Returns the first column(s) as data; last column as target
# Consider replacing this with pandas
def read_csv (file_name):
	import csv

	data = []
	targets = []

	with open(file_name, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		# Skip the first row. There may be a better way of doing this.
		header = True
		for row in lines:
			if header:
				header = False
			else:
				data.append(map(float, row[:-1]))
				targets.append(float(row[-1]))
	return data, targets


# Calculates and returns MSE
def calculate_mse (targets, hypotheses):
	mse = 0
	for i in range(len(targets)):
		mse += (targets[i] - hypotheses[i])**2
	return mse / len(targets)


# Returns the algorithm according to what's in opts
def get_algo (opts):
	from sklearn import linear_model
	from sklearn.neighbors import KNeighborsRegressor

	print 'Algorithm:', opts['algo']
	if opts['algo'].lower() == 'slr':
		return linear_model.LinearRegression()
	elif opts['algo'].lower() == 'knn':
		neighbours = 5     # Default to k=5
		# ... but take options from command line, if it's there.
		if 'algo_arg' in opts:
			neighbours = int(opts['algo_arg'])
		return KNeighborsRegressor(n_neighbors=neighbours)


# Drives the script by acting as a manager
def process_data (opts):
	import numpy as np
	from sklearn import linear_model

	train_x, train_y = read_csv(opts['train_file'])
	test_x, test_y = read_csv(opts['test_file'])
	# To view a plot of the data, uncomment the following:
	# plot(train_x, train_y)

	algo = get_algo(opts)
	if algo is not None:
        	algo.fit(train_x, train_y)
		hypotheses = algo.predict(test_x)
		print 'Test MSE:', calculate_mse(test_y, hypotheses)
	else:
		print 'Cannot generate predictions or MSE. (Are there options on command line?)'


# main

import sys
opts = get_args_by_position(sys.argv)
if len(opts) == 0:
	help()
else:
	process_data (opts)
