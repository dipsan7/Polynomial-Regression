import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("C:\\Users\\Dipsan\Desktop\\polynomial regression\\Position_Salaries.csv")
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
from sklearn.linear_model import  LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)
plt.scatter(X,y ,color="red")
plt.plot(X,lin_reg.predict(X),color="blue")
plt.title("level of salary")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()
plt.scatter(X,y ,color="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")
plt.title("level of salary")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()
#for predict of linear regression
lin_reg.predict([[6.5]])
#for predicting result with polynomial
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))