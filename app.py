import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from nltk.corpus import stopwords
import string
#string.punctuation
#nltk.download('stopwords')
#from sklearn.feature_extraction.text import TfidfVectorizer


def trans(text):
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



tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('mnb1_model.pkl','rb'))

st.title("Spam Classifier")
input_sms=st.text_input("Enter the message")

if st.button('Predict'):

    transform_sms=trans(input_sms)
    vector_input=tfidf.transform([transform_sms])
    result=model.predict(vector_input)[0]
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")