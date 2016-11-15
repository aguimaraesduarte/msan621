import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

djia = pd.read_table("djia.csv", sep=", ")
djia['Days'] = djia.index[::-1]
djia = djia[["Date","Days","Open","High","Low","Close"]]

X = djia["Days"]
Y = djia["Close"]

CV_indices = random.sample(xrange(0,len(X)), len(X)/10)
CV_X = [X[i] for i in CV_indices]
CV_Y = [Y[i] for i in CV_indices]

degrees = []
MSEs_train = []
MSEs_CV = []

for d in range(1,10):
	poly_params = np.polyfit(X, Y, d)
	Yfit = np.polyval(poly_params,X)
	SSE = np.sum((Y-Yfit)**2)
	SST = np.sum((Y-np.mean(Y))**2)
	MSE = SSE/len(Y)
	#Rsq = 1-SSE/SST
	#adjRsq = 1-((len(Y)-1.0)/(len(Y)-d))*(SSE/SST)
	Yfit_CV = np.polyval(poly_params,CV_X)
	SSE_CV = np.sum((CV_Y-Yfit_CV)**2)
	SST_CV = np.sum((CV_Y-np.mean(CV_Y))**2)
	MSE_CV = SSE_CV/len(CV_Y)
	degrees.append(d)
	MSEs_train.append(MSE)
	MSEs_CV.append(MSE_CV)

poly_params = np.polyfit(X, Y, 5)
plt.plot(X, Y, 'black')
plt.plot(X, np.polyval(poly_params, X), 'r-')
plt.show()
"""
plt.plot(degrees,MSEs_train,'black')#,label="train MSE")
plt.plot(degrees,MSEs_CV,'red')#,label="CV MSE")
plt.show()
"""

import numpy as np
from scipy.interpolate import UnivariateSpline

degree = 3 # In range 0 .. 5
smoothing = 1 # Lower bound for SSE

Xinv = X[::-1]
Yinv = Y[::-1]

s = UnivariateSpline(Xinv, Yinv, k=degree, s=4e9)

#get_coeffs(), get_knots(), get_residual()
plt.plot(Xinv, Yinv, "black")
plt.plot(Xinv, s(Xinv), "blue")
plt.show()

print len(s.get_knots())
print s.get_residual()