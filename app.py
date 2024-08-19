import string
import nltk
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import translate
# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Initialize translator
translator_translate = translate.Translator(to_lang='en', from_lang='auto')

# Load the data
df = pd.read_csv('2cls_spam_text_cls.csv')
messages = df['Message'].values.tolist()
labels = df['Category'].values.tolist()

# Text preprocessing function
def preprocess_txt(text):
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    stemmer = nltk.PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    return text

# Preprocess messages
messages = [preprocess_txt(message) for message in messages]

# Create vocabulary dictionary
def create_dict(messages):
    return list(dict.fromkeys(' '.join([' '.join(message) for message in messages]).split()))

word_dict = create_dict(messages)

# Create feature vector
def create_feature(message, word_dict):
    features = np.zeros(len(word_dict))
    for word in message:
        if word in word_dict:
            features[word_dict.index(word)] += 1
    return features

# Encode features and labels
X = np.array([create_feature(message, word_dict) for message in messages])
le = LabelEncoder()
y = le.fit_transform(labels)

# Split data
VALID_SIZE = 0.2
TEST_SIZE = 0.3
SEED = 0

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=VALID_SIZE, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=TEST_SIZE, random_state=SEED)

# Train model using Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluate model
y_valid_pred = model.predict(X_valid)
y_test_pred = model.predict(X_test)

acc_valid = accuracy_score(y_valid, y_valid_pred)
acc_test = accuracy_score(y_test, y_test_pred)

# Prediction function
def predict(message, model, word_dict):
    message = translator_translate.translate(message)
    message = preprocess_txt(message)
    features = create_feature(message, word_dict)
    features = np.array([features]).reshape(1, -1)
    prediction = model.predict(features)
    prediction_cls = le.inverse_transform(prediction)[0]
    return prediction_cls

# Streamlit app
st.title("Spam Classifier")
st.write("This is a simple spam classifier using Naive Bayes algorithm")

st.write(f"Validation accuracy: {acc_valid:.2f}")
st.write(f"Test accuracy: {acc_test:.2f}")

message = st.text_input("Enter message:")
if st.button("Predict"):
    prediction = predict(message, model, word_dict)
    st.write("Prediction: ", prediction)
