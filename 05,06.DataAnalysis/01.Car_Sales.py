import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('car_sales_new.csv')

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

print("How many values are 0? : ", df.price[df.price == 0].count())

df.columns = map(str.lower, df.columns)

df['drive'] = df['drive'].fillna("UnSpecified")
df.drive.unique()

df['engv'].fillna((df['engv'].mean()), inplace=True)

df['price'] = df['price'].replace(0, df['price'].mean())

mileage_avg = sum(df['mileage']) / len(df['mileage'])
df['mileage_level'] = ["high mileage" if i > mileage_avg else "low mileage" for i in df['mileage']]
df.loc[:10]

df['year'].value_counts().head(20).plot.bar()

df['body'].value_counts().head(20).plot.bar()
df['car'].value_counts().head(20).plot.bar()


usa = df.loc[df['car'] == 'Volkswagen']
top_10_model = usa['model'].value_counts()[:10].to_frame()
plt.figure(figsize=(10,5))

sns.barplot(x = top_10_model['model'], y = top_10_model.index, palette="PuBuGn_d")
plt.title('Top 10 Volkswagen model in terms of contribution', fontsize=18, fontweight="bold")
plt.xlabel('')
plt.show()




