# app.py
import streamlit as st
import pandas as pd
from customer_churn_prediction import run_churn_prediction_dashboard
from customer_sentiment_analysis import run_sentiment_analysis_dashboard
from product_recommendation import run_product_recommendation_dashboard

# Custom CSS for full-page flexbox layout
st.markdown(
    """
    <style>
    body {
        margin: 0;
        display: flex;
        flex-direction: column;
        height: 100vh;
    }
    .container {
        display: flex;
        flex: 1;
        padding: 20px;
    }
    .col {
        flex: 1;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title of the app
st.title("Customer Insights Dashboard")

# Sidebar for file uploads
st.sidebar.header("Upload your datasets")

# Initialize session state if not initialized
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = None
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = None
if 'transaction_data' not in st.session_state:
    st.session_state.transaction_data = None

customer_data_file = st.sidebar.file_uploader("Upload Customer Churn Data", type=["csv"])
feedback_data_file = st.sidebar.file_uploader("Upload Customer Feedback Data", type=["csv"])
transaction_data_file = st.sidebar.file_uploader("Upload Transaction Data", type=["csv"])

# Load datasets and display overview
if customer_data_file:
    st.session_state.customer_data = pd.read_csv(customer_data_file)
if feedback_data_file:
    st.session_state.feedback_data = pd.read_csv(feedback_data_file)
if transaction_data_file:
    st.session_state.transaction_data = pd.read_csv(transaction_data_file)

if st.session_state.customer_data is not None and st.session_state.feedback_data is not None and st.session_state.transaction_data is not None:
    st.sidebar.success("All files uploaded successfully!")

    # Function to run customer churn prediction dashboard
    @st.cache_data(hash_funcs={pd.DataFrame: lambda _: None})  # Disable caching for DataFrames
    def run_churn_dashboard(data):
        return run_churn_prediction_dashboard(data)

    # Function to run customer sentiment analysis dashboard
    @st.cache_data(hash_funcs={pd.DataFrame: lambda _: None})  # Disable caching for DataFrames
    def run_sentiment_dashboard(data):
        return run_sentiment_analysis_dashboard(data)

    # Function to run product recommendation dashboard
    @st.cache_data(hash_funcs={pd.DataFrame: lambda _: None})  # Disable caching for DataFrames
    def run_product_dashboard(data, customer_id):
        return run_product_recommendation_dashboard(data, customer_id)

    # Display customer churn prediction dashboard
    with st.container():
        with st.expander("Customer Churn Prediction"):
            st.subheader("Customer Churn Data Overview")
            st.write(st.session_state.customer_data.head())  # Display first few rows
            run_churn_dashboard(st.session_state.customer_data)

    # Display customer sentiment analysis dashboard
    with st.container():
        with st.expander("Customer Sentiment Analysis"):
            st.subheader("Customer Feedback Data Overview")
            st.write(st.session_state.feedback_data.head())  # Display first few rows
            run_sentiment_dashboard(st.session_state.feedback_data)

    # Display product recommendation dashboard
    with st.container():
        with st.expander("Product Recommendation"):
            st.subheader("Transaction Data Overview")
            st.write(st.session_state.transaction_data.head())  # Display first few rows

            # Get customer ID input for product recommendation
            customer_id = st.number_input("Enter Customer ID", min_value=1, max_value=1000, value=1)

            # Run product recommendation dashboard with selected customer ID
            run_product_dashboard(st.session_state.transaction_data, customer_id)

else:
    st.sidebar.warning("Please upload all three datasets to proceed.")
