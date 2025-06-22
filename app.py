import streamlit as st
import pickle
import string
import nltk

# Only needed once; can be commented out later
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# Text cleaning
def transform_text(text):
    text = text.lower()
    token = nltk.word_tokenize(text)

    y = [i for i in token if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model (1).pkl', 'rb'))

# Streamlit UI
st.title('📩 Email/SMS Spam Classifier')

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    # 1. Transform
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.error("🚨 Spam")
    else:
        st.success("✅ Not Spam")
