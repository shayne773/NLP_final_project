import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv('./data/test.csv')
corpus = df['article']
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)