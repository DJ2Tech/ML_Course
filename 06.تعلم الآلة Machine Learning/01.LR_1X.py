import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv('Advertising.csv')
X = df[['TV']]
Y = df['Sales']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=50)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train.values, Y_train)

b0 = round(model.intercept_,2)
b1 = round(model.coef_[0],2)


Tv_advertisment = [[120]]

pred = model.predict(Tv_advertisment)
print("prediction= ", pred[0])


Y_pred = model.predict(X_test.values)

diff = pd.DataFrame({'Actual':Y_test,'Predicted':Y_pred})
diff.head(10)

from sklearn import metrics
import numpy as np

mae = metrics.mean_absolute_error(Y_test,Y_pred)
mse = metrics.mean_squared_error(Y_test,Y_pred)
rmse = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))


print('mae =',mae.round(2))
print('mse =',mse.round(2))
print('rmse =',rmse.round(2))
