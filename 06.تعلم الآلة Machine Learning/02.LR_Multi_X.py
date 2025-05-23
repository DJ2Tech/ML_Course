import pandas as pd
import seaborn as sns
import sklearn

df = pd.read_csv('Advertising.csv')
x = df[['TV', 'Radio', 'Newspaper']]

y = df['Sales']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=15)
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_train.values,y_train)

print("The LR model is: Y = ",LR.intercept_, "+", LR.coef_[0], "TV + ", LR.coef_[1], "radio + ", LR.coef_[2], "newspaper")

y_pred = LR.predict(x_test.values)
sns.regplot(x = y_test, y = y_pred, color='green')

