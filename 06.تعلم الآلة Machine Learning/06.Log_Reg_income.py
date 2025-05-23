import pandas as pd

df = pd.read_csv('income.csv')

X = df[['Age','Education']]

y = df['VIP']

dfVIP=df[df['VIP']==1]

dfNotVIP=df[df['VIP']==0]


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X.values, y)

Age = 60

Education = 12

example = [[Age, Education]]

pred = model.predict(example)
if pred == 1:
    print("VIP")
else:
    print("Not VIP")
