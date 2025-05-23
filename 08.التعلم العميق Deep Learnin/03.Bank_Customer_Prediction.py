import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from joblib import dump


df = pd.read_csv('Churn_Modelling.csv')


input_columns = df.drop('Exited', axis=1)  

class_column = df['Exited']

oversampler = RandomOverSampler(random_state=0)

input_columns_resampled, class_column_resampled = oversampler.fit_resample(input_columns, class_column)


df_balanced = pd.concat([input_columns_resampled, class_column_resampled], axis=1)

class_distribution = df_balanced['Exited'].value_counts()


# Define the Input Data 
X = df_balanced.iloc[:, 3:13].values
# Define OutPut Data 
y = df_balanced.iloc[:, 13].values



labelencoder_gender = LabelEncoder()
X[:, 2] = labelencoder_gender.fit_transform(X[:, 2])
labelencoder_gender.transform(["Male","Female"])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)




X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
sc = StandardScaler()



# randome Seed to start the model wight and 
np.random.seed(42)
tf.random.set_seed(42)

model = Sequential()
input_dim = len(X_train[0])
model.add(Dense(6, activation = 'relu', input_dim = input_dim))

model.add(Dense(6, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid')) # Note Output is only 1 as we want one value to y , and we choose activation sigmoid as the output will be between 0 ,1  then we will make it 0 or 1 
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 10, epochs = 10)

evaluation = model.evaluate(X_test, y_test)
print("Loss:", evaluation[0])
print("Accuracy:", evaluation[1])



model.save("churn_model.h5")


dump(labelencoder_gender, "churn_label_encoder.pkl")


dump(ct, "churn_column_transformer.pkl")


dump(sc, "churn_standard_scaler.pkl")
