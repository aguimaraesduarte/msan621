import csv
import random

data = []
targets = []
file_name = "tweets.csv"
with open(file_name, 'rb') as f:
	lines = [line for line in f]

train = []
test = []
for line in lines:
	if random.random() < 0.2:
		test.append(line)
	else:
		train.append(line)

with open("tweets-train-2.csv", "w") as f:
	f.writelines(train)
with open("tweets-test-2.csv", "w") as f:
	f.writelines(test)