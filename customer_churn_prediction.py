#customer_churn_prediction.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt

def run_churn_prediction_dashboard(customer_data):
    st.header("Customer Churn Prediction")

    # Preprocess data
    X = customer_data.drop(columns=['customer_id', 'name', 'email', 'signup_date', 'last_purchase_date', 'churn'])
    y = customer_data['churn']
    
    # Encode categorical variables
    categorical_cols = ['location']  # List of categorical columns to encode
    
    def encode_categorical(df, col):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        return df
    
    # Use multiprocessing for encoding
    num_cores = multiprocessing.cpu_count()
    X_encoded = Parallel(n_jobs=num_cores)(delayed(encode_categorical)(X.copy(), col) for col in categorical_cols)
    X_encoded = X_encoded[0]  # Extract the encoded DataFrame from the list
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Define base models
    base_models = [
        ('xgb', XGBClassifier()),
        ('rf', RandomForestClassifier()),
        ('lr', LogisticRegression())
    ]
    
    # Define meta-model
    meta_model = LogisticRegression()
    
    # Create stacking ensemble
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
    
    # Train ensemble model
    stacking_model.fit(X_train, y_train)
    
    # Evaluate model
    stacking_auc = roc_auc_score(y_test, stacking_model.predict_proba(X_test)[:,1])
    st.write(f"Stacking Ensemble AUC: {stacking_auc:.2f}")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, stacking_model.predict_proba(X_test)[:,1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    st.pyplot(plt)

    # Predictions
    churn_predictions = stacking_model.predict_proba(X_encoded)[:,1]
    customer_data['churn_probability'] = churn_predictions
    st.subheader("Top Customers at Risk of Churn:")
    st.dataframe(customer_data[['customer_id', 'churn_probability']].sort_values(by='churn_probability', ascending=False).head(10))
