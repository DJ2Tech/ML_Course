import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


df = pd.read_csv('emails.csv')


nltk.download('punkt') 
nltk.download('stopwords')



# Make A Funcction to Process The Data 
def preprocess_text(text):
    
    tokens = word_tokenize(text.lower())
    
    tokens = [token for token in tokens if token not in string.punctuation]
    
    stop_words = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    preprocessed_text = ' '.join(tokens) # Join the Tokens to make it again as one text 

    preprocessed_text = re.sub(r'http\S+|www\S+', '', preprocessed_text) # Remove any hyper links form the Text 

    preprocessed_text = re.sub(r'\d+', '', preprocessed_text) # Remove all Numbers (\d+)
    return preprocessed_text


df['processed_Message'] = df['Message'].apply(preprocess_text)
df_spam = df[df['Spam']==1]




corpus = df['processed_Message']
max_features = 100
count_vectorizer = CountVectorizer(max_features=max_features)
vectors = count_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
tfidf_vectors = tfidf_vectorizer.fit_transform(corpus)




X = vectors
y = df['Spam']
X = X.toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)




model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)




y_pred_prob = model.predict(X_test)
y_pred = np.round(y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy*100,2))
