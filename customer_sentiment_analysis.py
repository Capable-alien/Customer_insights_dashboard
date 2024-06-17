#customer_sentiment_analysis.py
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt

def run_sentiment_analysis_dashboard(feedback_data):
    st.header("Customer Sentiment Analysis")

    # Load pre-trained sentiment-analysis model
    st.write("Loading sentiment-analysis model...")
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    st.write("Sentiment-analysis model loaded.")
    
    # Function to predict sentiment
    def predict_sentiment(text):
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Predict sentiment
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get predicted label
        sentiment_id = logits.argmax().item()
        sentiment_label = model.config.id2label[sentiment_id]
        
        return sentiment_label
    
    # Analyze sentiments
    feedback_data['sentiment'] = feedback_data['comment'].apply(predict_sentiment)
    
    # Display sentiment analysis results
    st.subheader("Sentiment Distribution:")
    sentiment_counts = feedback_data['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    st.subheader("Sample Sentiment Analysis Results:")
    st.dataframe(feedback_data[['customer_id', 'feedback', 'comment', 'sentiment']].sample(10))
