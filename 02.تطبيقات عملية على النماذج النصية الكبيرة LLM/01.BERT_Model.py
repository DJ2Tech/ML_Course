model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
from transformers import BertTokenizer, TFBertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name)

import pandas as pd
csv_file = 'IMDB Dataset.csv'
df = pd.read_csv(csv_file)




sentiment_mapping = {
    0: 'very negative',
    1: 'negative',
    2: 'neutral',
    3: 'positive',
    4: 'very positive'
}


first_5_rows_copy = first_5_rows.copy()
predicted_sentiments = []
for title in first_5_rows_copy['review']:
  inputs = tokenizer(title,)
  input_ids = inputs['input_ids']
  predictions = model.predict([input_ids])
  logits = predictions.logits
  predicted_class = np.argmax(logits)
  print("predicted id: ", predicted_class)
  predicted_sentiment = sentiment_mapping[predicted_class]
  predicted_sentiments.append(predicted_sentiment)


first_5_rows_copy['predicted_sentiment'] = predicted_sentiments
