import pandas as pd
import numpy as np
import sys
from sklearn import linear_model, metrics
from sklearn.ensemble import RandomForestRegressor

inp = sys.argv[1]
out = sys.argv[2]

file_life_exp = "life expectancy by country and year.csv"
file_gdp = "GDP by country and year.csv"
file_hiv = "HIV rates by country and year.csv"
file_income = "Income per capita by country and year.csv"
file_internet = "Internet user rates by country and year.csv"

years_1950_1960 = ["{}".format(i) for i in range(1950,1961)]
years_2010_2020 = ["{}".format(i) for i in range(2011,2021)]
pre = pd.DataFrame(np.nan, index=range(202), columns=years_1950_1960)
pos = pd.DataFrame(np.nan, index=range(202), columns=years_2010_2020)

life_exp = pd.read_csv(file_life_exp, header=0)
#life_exp = pd.concat([life_exp['Country Name'], life_exp[life_exp.columns[1:]].interpolate(method='linear', axis=1).ffill().bfill()], axis=1) #extrapolate linearly for missing life expectancy values

life_exp = pd.concat([life_exp['Country Name'], pd.concat([pre, life_exp[life_exp.columns[1:]], pos], axis=1).interpolate(method='linear', axis=1).ffill(axis=1).bfill(axis=1)], axis=1) #extrapolate linearly for missing life expectancy values
gdp = pd.read_csv(file_gdp, header=0)
gdp = pd.concat([gdp['Country Name'], pd.concat([pre, gdp[gdp.columns[1:]], pos], axis=1).fillna(0)], axis=1) #fill missing gdp with 0
hiv = pd.read_csv(file_hiv, header=0)
hiv = pd.concat([hiv['Country Name'], pd.concat([pre, hiv[hiv.columns[1:]], pos], axis=1).interpolate(method='linear', axis=1).ffill(axis=1)], axis=1).fillna(0)
income = pd.read_csv(file_income, header=0)
income = pd.concat([income['Country Name'], pd.concat([pre, income[income.columns[1:]], pos], axis=1).interpolate(method='linear', axis=1).ffill(axis=1)], axis=1).fillna(0)
internet = pd.read_csv(file_internet, header=0)
internet = pd.concat([internet['Country Name'], pd.concat([pre, internet[internet.columns[1:]], pos], axis=1).interpolate(method='linear', axis=1).ffill(axis=1)], axis=1).fillna(0)

# melt tables
life_exp = pd.melt(life_exp, id_vars='Country Name', var_name='Year', value_name='Life Expectancy')
gdp = pd.melt(gdp, id_vars='Country Name', var_name='Year', value_name='GDP Growth')
hiv = pd.melt(hiv, id_vars='Country Name', var_name='Year', value_name='HIV Rate')
income = pd.melt(income, id_vars='Country Name', var_name='Year', value_name='Income')
internet = pd.melt(internet, id_vars='Country Name', var_name='Year', value_name='Internet Use')

# create df
df = pd.merge(life_exp, gdp, "inner", ["Country Name","Year"])
df = pd.merge(df, hiv, "inner", ["Country Name","Year"])
df = pd.merge(df, income, "inner", ["Country Name","Year"])
df = pd.merge(df, internet, "inner", ["Country Name","Year"])

# create X, Y
X = df.drop(["Country Name", "Life Expectancy"], axis=1)
Y = df["Life Expectancy"]

# linear regression
model = linear_model.LinearRegression()
model.fit(X, Y)
#print model.coef_ #baseline coefficients [ 0.30481852 -0.04162589]

# random forest regression
estimators = 50
regressor = RandomForestRegressor(n_estimators=estimators)
regressor.fit(X, Y)

# read csv
def read_csv (file_name):
	import csv
	res = []
	with open(file_name, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		for row in lines:
			res.append(row)
	return res

# create predictions
test = read_csv(inp)
for row in test:
	if row[2] == "":
		row[2] = df[(df["Country Name"]==row[0]) & (df["Year"]==row[1])]["GDP Growth"].values[0]
	row.append(df[(df["Country Name"]==row[0]) & (df["Year"]==row[1])]["HIV Rate"].values[0])
	row.append(df[(df["Country Name"]==row[0]) & (df["Year"]==row[1])]["Income"].values[0])
	row.append(df[(df["Country Name"]==row[0]) & (df["Year"]==row[1])]["Internet Use"].values[0])

test = pd.DataFrame(test, columns=["Country Name", "Year", "GDP Growth", "HIV Rate", "Income", "Internet Use"])
test = test.drop(test.columns[0], axis=1)
hypotheses = regressor.predict(test)
pd.DataFrame(hypotheses, columns=["Prediction"]).to_csv(out, index=False, header=False)


"""
test_countries = ["Vietnam"]

# test and train
train = df.loc[~df['Country Name'].isin(test_countries)].reset_index(drop=True)
test = df.loc[df['Country Name'].isin(test_countries)].reset_index(drop=True)

X_train = train.drop(["Country Name", "Life Expectancy"], axis=1)
Y_train = train["Life Expectancy"]
X_test = test.drop(["Country Name", "Life Expectancy"], axis=1)
Y_test = test["Life Expectancy"]

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
print model.coef_ #baseline coefficients
hypotheses = model.predict(X_test)
print "MSE: %.3f" %metrics.mean_squared_error(Y_test, hypotheses)

# forest
estimators = 10
regressor = RandomForestRegressor(n_estimators=estimators)
regressor.fit(X_train, Y_train)
hypotheses = regressor.predict(X_test)
print "MSE: %.3f" %metrics.mean_squared_error(Y_test, hypotheses)
"""