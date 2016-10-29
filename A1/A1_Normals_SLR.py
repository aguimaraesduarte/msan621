from sklearn import linear_model, metrics
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
import sys

import time
t = time.time()

if len(sys.argv) == 1:
	print "Need test stations.\nex: python A1_Normals.py USW00023234 USW00014918"
	sys.exit()
else:
	test_IDs = sys.argv[1:]
#test_IDs = ["USW00023234", "USW00014918", "USW00012919", "USW00013743", "USW00025309"]

hours = ["H00", "H01", "H02", "H03", "H04", "H05", "H06", "H07", "H08", "H09", "H10", "H11", "H12",
         "H13", "H14", "H15", "H16", "H17", "H18", "H19", "H20", "H21", "H22", "H23"]
ids = ["ID", "Month", "Day"]
names = ids + hours
normals = pd.read_table("hly-temp-normal.txt", sep="\s+", header=None, names=names)

num_stations_test = len(test_IDs)
num_stations_train = 457 - num_stations_test

#print "Data read."

# split into train and test sets
train = normals.loc[~normals['ID'].isin(test_IDs)].reset_index(drop=True)
test = normals.loc[normals['ID'].isin(test_IDs)].reset_index(drop=True)

# separate hourly data from rest for removing trailing character
temps_train = train.drop(train.columns[range(3)], axis=1)
IDs_train = train.drop(train.columns[range(3,27)], axis=1)
temps_test = test.drop(test.columns[range(3)], axis=1)
IDs_test = test.drop(test.columns[range(3,27)], axis=1)

# remove trailing character
for i in range(len(temps_train.columns)):
	temps_train[temps_train.columns[range(len(temps_train.columns))][i]] = temps_train[temps_train.columns[range(len(temps_train.columns))][i]].astype(str).str[:-1]
	temps_test[temps_test.columns[range(len(temps_test.columns))][i]] = temps_test[temps_test.columns[range(len(temps_test.columns))][i]].astype(str).str[:-1]

#print "Removed trailing chars."

# perform imputation
imp = Imputer(missing_values=-999, strategy='mean', axis=0)
temps_train = imp.fit_transform(temps_train)
temps_test = imp.fit_transform(temps_test)

#print "Imputation performed."

# join back into pandas dataframe
train = pd.concat([IDs_train, pd.DataFrame(temps_train)], axis=1)
train.columns = names
test = pd.concat([IDs_test, pd.DataFrame(temps_test)], axis=1)
test.columns = names

#print "Joined into single df."

train = pd.melt(train, id_vars=ids, value_vars=hours, var_name='Hour', value_name='Temp').sort_values(['ID','Month','Day','Hour']).reset_index(drop=True)
test = pd.melt(test, id_vars=ids, value_vars=hours, var_name='Hour', value_name='Temp').sort_values(['ID','Month','Day','Hour']).reset_index(drop=True)

#print "Tables melted."

# previous hour temp
previousHourTemp_train = list(train.groupby(['ID']).apply(lambda x: x.reindex(index = np.roll(x.index, 1)))['Temp'].reset_index(drop=True))
previousHourTemp_test = list(test.groupby(['ID']).apply(lambda x: x.reindex(index = np.roll(x.index, 1)))['Temp'].reset_index(drop=True))

#print "Feature 1 calculated."

# previous day temp
previousDayTemp_train = list(train.groupby(['ID']).apply(lambda x: x.reindex(index = np.roll(x.index, 24)))['Temp'].reset_index(drop=True))
previousDayTemp_test = list(test.groupby(['ID']).apply(lambda x: x.reindex(index = np.roll(x.index, 24)))['Temp'].reset_index(drop=True))

#print "Feature 2 calculated."

# mean per day per hour
meanPerDayPerHour_train = list(train.groupby(['Month','Day','Hour'])['Temp'].mean()) * num_stations_train
meanPerDayPerHour_test = list(test.groupby(['Month','Day','Hour'])['Temp'].mean()) * num_stations_test

#print "Feature 3 calculated."

# mean that day up to but not including current hour
# note: for H00, the temp is set as that particular temp, since no previous value exists for that day
runningMean_train = [temp for runningMeanPerDayPerStation in [[np.mean(train['Temp'].values[i:(i + end)]) if end > 0 else train['Temp'][i] for end in range(24)] for i in range(0, len(train), 24)] for temp in runningMeanPerDayPerStation]
runningMean_test = [temp for runningMeanPerDayPerStation in [[np.mean(test['Temp'].values[i:(i + end)]) if end > 0 else test['Temp'][i] for end in range(24)] for i in range(0, len(test), 24)] for temp in runningMeanPerDayPerStation]

#print "Feature 4 calculated."

# create train and test sets
train_x = pd.DataFrame({"prevHour":previousHourTemp_train, "prevDay":previousDayTemp_train, "meanTemp":meanPerDayPerHour_train, "runningMean":runningMean_train})
test_x = pd.DataFrame({"prevHour":previousHourTemp_test, "prevDay":previousDayTemp_test, "meanTemp":meanPerDayPerHour_test, "runningMean":runningMean_test})
train_y = train['Temp']
test_y = test['Temp']

#print "Tests created"

# algorithm
algo = linear_model.LinearRegression()
algo.fit(train_x, train_y)
hypotheses = algo.predict(test_x)

print "MSE: %.3f" % metrics.mean_squared_error(test_y, hypotheses)