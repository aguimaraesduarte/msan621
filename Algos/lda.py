

# Prints some "how to use this script" instructions
def help ():
	print 'Arguments by position:'
	print '\t1: Data file - Full or relative path; data will be split into 80/20 train/test'


# Populates and returns a dictionary with options from command line
# Replace this with argparse (or getopt)
# cf. https://docs.python.org/2/howto/argparse.html
def get_args_by_position (args):
	opts = {}

	if len(args) != 2:
		return opts
	opts['data_file'] = args[1]

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
				data.append(map(float, row[1:]))
				targets.append(float(row[0]))
	return data, targets


# Returns the algorithm according to what's in opts
def get_algo (opts):
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

	# Include priors, ala below for change to posteriors. Priors must sum to 1.0
	# return LinearDiscriminantAnalysis(priors=[0.2,0.8]) 
	return LinearDiscriminantAnalysis()


# Prints the classification error
def print_accuracy (hypotheses, targets):
	total = 0
	hits = 0

	for i in range(len(hypotheses)):
		total += 1
		if hypotheses[i] == targets[i]:
			hits += 1
	print 'Classification Error:', float(total-hits) / float(total)


# Drives the script by acting as a manager
def process_data (opts):
	data, targets = read_csv(opts['data_file'])
	split = int(len(data) * 0.8)
	print 'Splitting data: training on', split+1, 'instances; testing on', len(data)-split, 'instances.'
	train_x = data[:split]
	train_y = targets[:split]
	test_x = data[split:]
	test_y = targets[split:]
	# To view a plot of the data, uncomment the following:
	# plot(train_x, train_y)

	algo = get_algo(opts)
	if algo is not None:
        	algo.fit(train_x, train_y)
		hypotheses = algo.predict(test_x)
		print_accuracy(hypotheses, test_y)
		# print 'Classification accuracy:', algo.score(test_x, test_y) * 100.0
	else:
		print 'Cannot generate predictions or MSE. (Are there options on command line?)'


# main

import sys
opts = get_args_by_position(sys.argv)
if len(opts) == 0:
	help()
else:
	process_data (opts)
