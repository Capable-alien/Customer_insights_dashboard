# product_recommendation.py
import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic
import matplotlib.pyplot as plt

def run_product_recommendation_dashboard(transaction_data, customer_id):
    st.header("Product Recommendation")

    # Load transaction data
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(transaction_data[['customer_id', 'product_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    # Train models
    algo_svd = SVD()
    algo_knn = KNNBasic()
    
    algo_svd.fit(trainset)
    algo_knn.fit(trainset)
    
    # Function to get recommendations for a customer
    def get_recommendations(customer_id, top_n=5):
        items = trainset.all_items()
        svd_preds = [algo_svd.predict(customer_id, item).est for item in items]
        knn_preds = [algo_knn.predict(customer_id, item).est for item in items]
        
        # Average predictions
        ensemble_preds = [(item, (svd_preds[i] + knn_preds[i]) / 2) for i, item in enumerate(items)]
        ensemble_preds.sort(key=lambda x: x[1], reverse=True)
        return ensemble_preds[:top_n]
    
    # Get recommendations for selected customer ID
    recommendations = get_recommendations(customer_id)
    
    # Display recommendations
    st.subheader(f"Top {len(recommendations)} Recommendations for Customer {customer_id}:")
    recommended_df = pd.DataFrame(recommendations, columns=['Product ID', 'Estimated Rating'])
    st.dataframe(recommended_df)
