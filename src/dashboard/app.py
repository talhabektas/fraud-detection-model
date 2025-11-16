"""
Real-Time Fraud Detection Dashboard
Streamlit app for monitoring fraud detection system
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient
from datetime import datetime, timedelta
import time
import numpy as np


# Page config
st.set_page_config(
    page_title="🚨 Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


class FraudDashboard:
    """Dashboard for monitoring fraud detection system"""
    
    def __init__(self, mongodb_uri='mongodb://admin:fraudadmin123@localhost:27017'):
        """Initialize dashboard with MongoDB connection"""
        self.mongodb_uri = mongodb_uri
        self.client = None
        self.db = None
        self.collection = None
        
    def connect_mongodb(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.mongodb_uri)
            self.db = self.client['fraud_detection']
            self.collection = self.db['predictions']
            # Test connection
            self.client.server_info()
            return True
        except Exception as e:
            st.error(f"❌ MongoDB Connection Error: {e}")
            return False
    
    def get_recent_predictions(self, limit=1000):
        """Get recent predictions from MongoDB"""
        try:
            cursor = self.collection.find().sort('processing_time', -1).limit(limit)
            df = pd.DataFrame(list(cursor))
            
            if len(df) > 0 and '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            return df
        except Exception as e:
            st.error(f"Error fetching predictions: {e}")
            return pd.DataFrame()
    
    def get_stats(self, df):
        """Calculate statistics from predictions"""
        if len(df) == 0:
            return {
                'total': 0,
                'fraud_detected': 0,
                'fraud_rate': 0,
                'avg_amount': 0,
                'fraud_amount': 0,
                'high_risk': 0
            }
        
        stats = {
            'total': len(df),
            'fraud_detected': int((df['prediction'] == 1).sum()),
            'fraud_rate': (df['prediction'] == 1).mean() * 100,
            'avg_amount': df['Amount'].mean(),
            'fraud_amount': df[df['prediction'] == 1]['Amount'].sum(),
            'high_risk': int((df['fraud_probability'] > 0.7).sum())
        }
        
        # Accuracy if we have labels
        if 'is_fraud' in df.columns and df['is_fraud'].notna().any():
            stats['accuracy'] = (df['correct_prediction'].sum() / len(df)) * 100
        else:
            stats['accuracy'] = None
        
        return stats


def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<div class="main-header">🚨 Real-Time Fraud Detection Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # MongoDB URI
    mongodb_uri = st.sidebar.text_input(
        "MongoDB URI", 
        value="mongodb://admin:fraudadmin123@localhost:27017"
    )
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 60, 5)
    
    # Limit
    data_limit = st.sidebar.number_input("Data Limit", min_value=100, max_value=10000, 
                                         value=1000, step=100)
    
    # Initialize dashboard
    dashboard = FraudDashboard(mongodb_uri)
    
    # Connect to MongoDB
    if not dashboard.connect_mongodb():
        st.error("❌ Failed to connect to MongoDB. Please check your connection settings.")
        st.stop()
    
    st.sidebar.success("✅ Connected to MongoDB")
    
    # Fetch data
    with st.spinner("📊 Loading data..."):
        df = dashboard.get_recent_predictions(limit=data_limit)
    
    if len(df) == 0:
        st.warning("⚠️ No data available yet. Start the producer and consumer to see live data.")
        st.stop()
    
    # Calculate stats
    stats = dashboard.get_stats(df)
    
    # Display metrics
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Transactions", f"{stats['total']:,}")
    
    with col2:
        st.metric("Fraud Detected", f"{stats['fraud_detected']:,}", 
                 delta=f"{stats['fraud_rate']:.2f}%")
    
    with col3:
        st.metric("High Risk (>70%)", f"{stats['high_risk']:,}")
    
    with col4:
        st.metric("Avg Amount", f"${stats['avg_amount']:.2f}")
    
    with col5:
        if stats['accuracy'] is not None:
            st.metric("Accuracy", f"{stats['accuracy']:.2f}%")
        else:
            st.metric("Fraud Amount", f"${stats['fraud_amount']:,.2f}")
    
    # Charts
    st.markdown("---")
    
    # Row 1: Fraud Distribution & Amount Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🥧 Fraud vs Normal")
        fraud_counts = df['prediction'].value_counts()
        fig = px.pie(
            values=fraud_counts.values,
            names=['Normal', 'Fraud'],
            color=['Normal', 'Fraud'],
            color_discrete_map={'Normal': '#4CAF50', 'Fraud': '#F44336'},
            hole=0.4
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💰 Transaction Amount Distribution")
        fig = px.histogram(
            df, 
            x='Amount', 
            color='prediction',
            nbins=50,
            color_discrete_map={0: '#4CAF50', 1: '#F44336'},
            labels={'prediction': 'Type', 0: 'Normal', 1: 'Fraud'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 2: Fraud Probability & Timeline
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Fraud Probability Distribution")
        fig = px.histogram(
            df,
            x='fraud_probability',
            nbins=50,
            color='prediction',
            color_discrete_map={0: '#4CAF50', 1: '#F44336'}
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                     annotation_text="Threshold (0.5)")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### ⏰ Fraud Detection Over Time")
        if 'processing_time' in df.columns:
            df_time = df.copy()
            df_time['processing_time'] = pd.to_datetime(df_time['processing_time'])
            df_time = df_time.sort_values('processing_time')
            df_time['cumulative_fraud'] = (df_time['prediction'] == 1).cumsum()
            
            fig = px.line(
                df_time,
                x='processing_time',
                y='cumulative_fraud',
                color_discrete_sequence=['#F44336']
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent Fraud Alerts
    st.markdown("---")
    st.markdown("### 🚨 Recent Fraud Alerts")
    
    fraud_df = df[df['prediction'] == 1].sort_values('fraud_probability', ascending=False).head(10)
    
    if len(fraud_df) > 0:
        display_cols = ['transaction_id', 'Amount', 'fraud_probability', 'processing_time']
        display_cols = [col for col in display_cols if col in fraud_df.columns]
        
        fraud_display = fraud_df[display_cols].copy()
        
        if 'fraud_probability' in fraud_display.columns:
            fraud_display['fraud_probability'] = fraud_display['fraud_probability'].apply(
                lambda x: f"{x:.2%}"
            )
        if 'Amount' in fraud_display.columns:
            fraud_display['Amount'] = fraud_display['Amount'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(fraud_display, use_container_width=True)
    else:
        st.info("No fraud detected in recent transactions.")
    
    # Confusion Matrix (if labels available)
    if 'is_fraud' in df.columns and df['is_fraud'].notna().any():
        st.markdown("---")
        st.markdown("### 📊 Model Performance (Confusion Matrix)")
        
        from sklearn.metrics import confusion_matrix
        
        y_true = df['is_fraud'].dropna()
        y_pred = df.loc[y_true.index, 'prediction']
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Normal', 'Fraud'],
            y=['Normal', 'Fraud'],
            color_continuous_scale='Reds',
            text_auto=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
