import pandas as pd
import numpy as np
import sys
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

if len(sys.argv) != 3:
	print 'Arguments by position:'
	print '\t1: Input file    - Full or relative path'
	print '\t2: Output file   - Full or relative path'
	sys.exit(0)

inp = sys.argv[1]
out = sys.argv[2]

# files
file_life_exp = "life expectancy by country and year.csv"

# missing chunks of data
years_1950_1960 = ["{}".format(i) for i in range(1950,1961)]
years_2010_2016 = ["{}".format(i) for i in range(2011,2017)]
pre = pd.DataFrame(np.nan, index=range(202), columns=years_1950_1960)
pos = pd.DataFrame(np.nan, index=range(202), columns=years_2010_2016)

# read in data
life_exp = pd.read_csv(file_life_exp, header=0)

# country names
countries = life_exp['Country Name']

# add columns for 1950-1960 and 2010-2016
life_exp = pd.concat([pre, life_exp[life_exp.columns[1:]], pos], axis=1)

# inpute missing data
life_exp = life_exp.interpolate(method='linear', axis=1).bfill(axis=1).ffill(axis=1).fillna(0)

# encode country labels
le = preprocessing.LabelEncoder()
le.fit(countries)
countries_encoded = le.transform(countries)
countries_encoded = pd.DataFrame(countries_encoded, columns=["Country Label"])

# put country names (encoded) back in
life_exp = pd.concat([countries_encoded, life_exp], axis=1)

# melt tables
life_exp = pd.melt(life_exp, id_vars='Country Label', var_name='Year', value_name='Life Expectancy')

# make Year columns int64
life_exp["Year"] = pd.to_numeric(life_exp["Year"])

# create X, Y
X = life_exp[life_exp.columns[:2]]
Y = life_exp["Life Expectancy"]

# random forest regression
estimators = 50
regressor = RandomForestRegressor(n_estimators=estimators)
regressor.fit(X, Y)

# read input file
test_df = pd.read_csv(inp, header=None, names=["Country Name","Year","GDP"])
test_countries = test_df["Country Name"]
test_countries_encoded = le.transform(test_countries)
test_countries_encoded = pd.DataFrame(test_countries_encoded, columns=["Country Label"])
test_X = pd.concat([test_countries_encoded, test_df["Year"]], axis=1)

# construct missing columns for test X
n = test_df.shape[0]
v_countries = []
v_years = []
for i in range(n):
	country_encoded = test_X.iloc[i][0]
	year = test_X.iloc[i][1]
	v_countries.append(test_X.iloc[i][0])
	v_years.append(test_X.iloc[i][1])
test_X = pd.DataFrame([v_countries, v_years]).transpose()
test_X.columns = X.columns
test_X["Year"] = test_X["Year"].astype(int)

# create predictions
hypotheses = regressor.predict(test_X)
pd.DataFrame(hypotheses, columns=["Prediction"]).to_csv(out, index=False, header=False)
