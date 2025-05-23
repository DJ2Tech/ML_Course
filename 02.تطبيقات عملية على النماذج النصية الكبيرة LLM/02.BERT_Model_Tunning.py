model_name = 'google-bert/bert-base-uncased'


from transformers import BertTokenizer, TFBertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name)

import pandas as pd
df = pd.read_csv(file_path, sep='\t')


import numpy as np

sentiment_mapping = {
    0: 'mixed',
    1: 'negative',
    2: 'positive',
}

random_5_row = random_5_row.copy()

predicted_sentiments = []

for title in random_5_row['text']:

    inputs = tokenizer(title,)

    input_ids = inputs['input_ids']

    predictions = model.predict([input_ids])

    logits = predictions.logits

    predicted_class = np.argmax(logits)
    print("predicted id: ", predicted_class)

    print(predictions)

    predicted_sentiment = sentiment_mapping[predicted_class]
    predicted_sentiments.append(predicted_sentiment)

random_5_row['predicted_sentiment'] = predicted_sentiments


from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
df['label'] = LE.fit_transform(df['label'])



from sklearn.model_selection import train_test_split
train, val = train_test_split(df, test_size=0.2, random_state=42)

train.reset_index(drop=True, inplace=True)
val.reset_index(drop=True, inplace=True)


x_train = train['text'].to_numpy()
y_train = train['label'].to_numpy()
x_test = val['text'].to_numpy()
y_test = val['label'].to_numpy()


import ktrain
from ktrain import text
MODEL_NAME = 'bert-base-uncased'
categories = ['Mixed','Negative','Positive']
t = text.Transformer(MODEL_NAME, maxlen=200, class_names= categories)
trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_test, y_test)

model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)

import time
start = time.time()
learner.fit_onecycle(2e-5, 4)
stop = time.time()
print(f"Training time: {stop - start}s")

learner.validate(class_names= categories)

predictor = ktrain.get_predictor(learner.model, preproc=t).save('/content/drive/My Drive/03. path/Finetune_BERT/Finetuned_Arabic_Sentiment_BERT')


predictor = ktrain.load_predictor('/content/drive/My Drive/03. path/Finetune_BERT/Finetuned_Arabic_Sentiment_BERT')
model = ktrain.get_predictor(predictor.model, predictor.preproc)

x = ["كله رائع بجد ربنا يكرمك", "اتقوا الله فينا بكفي رفع اسعار الرواتب بالحضيض"]
predictions = model.predict(x)

print(predictions)














