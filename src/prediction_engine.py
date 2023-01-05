import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class PredictionModel:
    def __init__(self):
        nltk.download('stopwords')
        news_dataset = pd.read_csv('train.csv')
        news_dataset = news_dataset.fillna('')
        port_stem = PorterStemmer()

        def stemming(content):
            stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
            stemmed_content = stemmed_content.lower()
            stemmed_content = stemmed_content.split()
            stemmed_content = [port_stem.stem(word) for word in stemmed_content if
                               not word in stopwords.words('english')]
            stemmed_content = ' '.join(stemmed_content)
            return stemmed_content

        news_dataset['title'] = news_dataset['title'].apply(stemming)

        X = news_dataset['title'].values
        Y = news_dataset['label'].values

        vectorizer = TfidfVectorizer()
        vectorizer.fit(X)

        X = vectorizer.transform(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

        model = LogisticRegression()
        model.fit(X_train, Y_train)

        X_train_prediction = model.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
        print('Accuracy score of the training data : ', training_data_accuracy)

        X_test_prediction = model.predict(X_test)
        test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
        print('Accuracy score of the test data : ', test_data_accuracy)

        self.vectorizer = vectorizer
        self.model = model

    def predict(self, text) -> float:
        text = self.vectorizer.transform([text])
        prediction = self.model.predict_proba(text)
        return prediction[0][0]
