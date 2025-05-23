import pandas as pd
from sklearn import preprocessing
df = pd.read_csv('cars.csv')


le_brand = preprocessing.LabelEncoder()
le_brand.fit(df['Brand']) 

le_body = preprocessing.LabelEncoder()
le_body.fit(df['Body']) 

le_model = preprocessing.LabelEncoder()
le_model.fit(df['Model']) 

le_enginetype = preprocessing.LabelEncoder()
le_enginetype.fit(df['Engine Type']) 

le_reg = preprocessing.LabelEncoder()
le_reg.fit(df['Registration']) 


# add data to dataframe
df['Brand_num'] = le_brand.transform(df['Brand']) 
df['Body_num'] = le_body.transform(df['Body']) 
df['Model_num'] = le_model.transform(df['Model']) 
df['EngineType_num'] = le_enginetype.transform(df['Engine Type']) 
df['Registration_num'] = le_reg.transform(df['Registration']) 

# drop the old Columns 
df=df.drop(columns=['Brand','Body','Engine Type','Registration','Model'])

df['EngineV'].fillna(value=df['EngineV'].mean(), inplace=True)


X = df[['Year','Mileage' ,'Body_num','Brand_num']]
y = df['Price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train.values, y_train)

y_pred = model.predict(X_test.values)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

print ('MSE :', mean_squared_error (y_pred, y_test))
print ('RMSE :', sqrt(mean_squared_error (y_pred, y_test)))
print ('MAE :', mean_absolute_error(y_pred, y_test))



