import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


tfid = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title("Spam Classifier")
input_sms = st.text_input("Enter your SMS")

if st.button("Predict"):

    transformed_text = transform_text(input_sms)
    vector = tfid.transform([transformed_text])
    result = model.predict(vector)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")