# Customer Insights Dashboard

The Customer Insights Dashboard is an interactive Streamlit application designed to help businesses gain valuable insights from their customer data. This dashboard offers three key functionalities:

1. **Customer Churn Prediction**: Utilize machine learning to predict which customers are most likely to churn, allowing you to take proactive measures to retain them.
2. **Customer Sentiment Analysis**: Analyze customer feedback to understand their sentiments and identify areas for improvement in your products or services.
3. **Product Recommendation**: Provide personalized product recommendations to customers based on their past transaction data, enhancing customer satisfaction and driving sales.

## Features

- **Customer Churn Prediction**: 
  - Preprocess customer data and train a stacking ensemble model combining XGBoost, RandomForest, and Logistic Regression.
  - Visualize the ROC curve and display the top customers at risk of churn.

- **Customer Sentiment Analysis**:
  - Use a pre-trained BERT model to classify customer feedback into different sentiment categories.
  - Display sentiment distribution and sample analysis results.

- **Product Recommendation**:
  - Train collaborative filtering models (SVD and KNN) on transaction data.
  - Generate and display top product recommendations for each customer.

## Setup

1.  **Install dependencies** : `pip install -r requirements.txt`
2.  **Run the Streamlit app** : `streamlit run app.py`

## Usage

1.  **Upload your datasets via the sidebar**:
      -Customer Churn Data: Contains customer information and churn status.
      -Customer Feedback Data: Contains customer feedback comments.
      -Transaction Data: Contains customer transaction history and ratings.
    
2.  **Navigate through the different sections of the dashboard to gain insights:
      -Customer Churn Prediction: View and analyze the risk of customer churn.
      -Customer Sentiment Analysis: Understand customer sentiments from their feedback.
      -Product Recommendation: Get personalized product recommendations for customers.

**Ensure your datasets are in CSV format with appropriate columns. Check the example datasets provided in the `/dataset` folder.**
