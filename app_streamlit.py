import streamlit as st
import joblib

# Load pre-trained models and vectorizers
sentiment_model = joblib.load('models/sentiment_model.pkl')
lda_model = joblib.load('models/lda_model.pkl')
sentiment_vectorizer = joblib.load('models/vectorizer.pkl')
topic_vectorizer = joblib.load('models/tf_vectorizer.pkl')

# Function to predict sentiment and topic
def predict(text):
    # Predict sentiment
    text_vec = sentiment_vectorizer.transform([text])
    sentiment = sentiment_model.predict(text_vec)
    
    # Predict topic
    topic_vec = topic_vectorizer.transform([text])
    topic_distribution = lda_model.transform(topic_vec)
    topic = topic_distribution.argmax()
    
    return sentiment[0], topic

# Streamlit App
st.title("NLP Analysis on SMS Dataset")

message = st.text_area("Enter your message here", "")
if st.button("Predict"):
    sentiment, topic = predict(message)
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Topic: {topic}")

if __name__ == "__main__":
    st.write("NLP Analysis with Streamlit")
