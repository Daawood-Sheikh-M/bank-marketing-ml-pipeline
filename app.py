import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)

# Set page configuration
st.set_page_config(
    page_title="Bank Marketing ML Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        padding-bottom: 10px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F8FAFC;
        border-radius: 8px 8px 0px 0px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏦 Bank Marketing Term Deposit Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">BITS Pilani M.Tech Data Science & Engineering | End-to-End Machine Learning Pipeline</div>', unsafe_allow_html=True)

# Helper function to load artifacts
@st.cache_resource
def load_artifacts():
    models = {}
    model_files = {
        "Logistic Regression": "models/logistic.pkl",
        "Decision Tree": "models/decision_tree.pkl",
        "KNN": "models/knn.pkl",
        "Naive Bayes": "models/naive_bayes.pkl",
        "Random Forest": "models/random_forest.pkl"
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
                    
    preprocessor = None
    scaler = None
    if os.path.exists("models/preprocessor.pkl"):
        with open("models/preprocessor.pkl", 'rb') as f:
            preprocessor = pickle.load(f)
            
    if os.path.exists("models/scaler.pkl"):
        with open("models/scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
            
    return models, preprocessor, scaler

models, preprocessor, scaler = load_artifacts()

# Helper function to load test dataset
@st.cache_data
def load_test_data():
    if os.path.exists("data/test_data.csv"):
        return pd.read_csv("data/test_data.csv")
    return None

test_df = load_test_data()

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Exploratory Data Analysis", 
    "📈 Model Evaluation Overview", 
    "🔮 Interactive Prediction Engine"
])

# ==========================================
# TAB 1: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab1:
    st.header("📊 Exploratory Data Analysis & Visual Insights")
    
    if test_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{len(test_df):,}")
        col2.metric("Total Features", f"{test_df.shape[1] - 1}")
        col3.metric("Numerical Features", "7")
        col4.metric("Categorical Features", "9")
        
        st.markdown("---")
        
        # Row 1 EDA Charts
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Target Distribution (Deposit Subscription)")
            fig, ax = plt.subplots(figsize=(6, 4))
            target_counts = test_df['deposit'].value_counts()
            ax.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', colors=['#3B82F6', '#10B981'], startangle=140, explode=(0.05, 0))
            ax.set_title("Class Proportion", fontweight='bold')
            st.pyplot(fig)
            st.info("💡 **Inference:** The target classes are well-balanced (~52.6% No vs ~47.4% Yes), ensuring stable model convergence without needing synthetic oversampling.")

        with c2:
            st.subheader("Contact Duration vs Deposit Outcome")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x='deposit', y='duration', data=test_df, palette='Set2', ax=ax)
            ax.set_title("Call Duration (seconds) distribution by Class", fontweight='bold')
            st.pyplot(fig)
            st.info("💡 **Inference:** Successful deposit subscriptions show significantly longer call durations. Call duration is one of the strongest predictive features.")

        st.markdown("---")
        
        # Row 2 EDA Charts
        c3, c4 = st.columns(2)
        
        with c3:
            st.subheader("Job Type vs Subscription Count")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.countplot(y='job', hue='deposit', data=test_df, palette='crest', order=test_df['job'].value_counts().index, ax=ax)
            ax.set_title("Subscription by Job Category", fontweight='bold')
            st.pyplot(fig)
            st.info("💡 **Inference:** Management, Technicians, and Admin roles represent the largest conversion volume, while students and retirees have higher relative success ratios.")

        with c4:
            st.subheader("Numerical Feature Correlation Matrix")
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            num_df = test_df.select_dtypes(include=['int64', 'float64'])
            sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='Blues', linewidths=0.5, ax=ax)
            ax.set_title("Masked Correlation Matrix", fontweight='bold')
            st.pyplot(fig)
            st.info("💡 **Inference:** Low pairwise correlations across numerical attributes indicate minimal severe multicollinearity, ensuring stable linear estimations.")
    else:
        st.warning("⚠️ Dataset not found in `data/test_data.csv`.")

# ==========================================
# TAB 2: MODEL PERFORMANCE OVERVIEW
# ==========================================
with tab2:
    st.header("📈 Pre-trained Models Performance Evaluation")
    
    if test_df is not None and len(models) > 0 and preprocessor is not None and scaler is not None:
        target_col = 'deposit' if 'deposit' in test_df.columns else test_df.columns[-1]
        X_test_raw = test_df.drop(columns=[target_col])
        y_test = test_df[target_col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)
        
        X_test_proc = preprocessor.transform(X_test_raw)
        X_test_scaled = scaler.transform(X_test_proc)
        
        eval_results = []
        model_cm = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test_scaled)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_prob = model.decision_function(X_test_scaled)
                
            eval_results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_prob),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                "MCC Score": matthews_corrcoef(y_test, y_pred)
            })
            model_cm[name] = confusion_matrix(y_test, y_pred)
            
        metrics_df = pd.DataFrame(eval_results)
        
        # Display top model banner
        best_model_name = metrics_df.sort_values(by="MCC Score", ascending=False).iloc[0]["Model"]
        best_accuracy = metrics_df.sort_values(by="MCC Score", ascending=False).iloc[0]["Accuracy"]
        st.success(f"🏆 **Top Performing Model:** `{best_model_name}` (Accuracy: {best_accuracy:.4f} | Highest MCC Score)")
        
        st.dataframe(
            metrics_df.style.highlight_max(axis=0, color='#D1FAE5'),
            use_container_width=True
        )
        
        st.markdown("---")
        st.subheader("🔍 Confusion Matrix Comparison")
        selected_eval_model = st.selectbox("Choose Model to Inspect Matrix:", list(models.keys()))
        
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(model_cm[selected_eval_model], annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 12, "weight": "bold"})
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix - {selected_eval_model}", fontweight='bold')
        st.pyplot(fig)
        
    else:
        st.warning("⚠️ Model artifacts missing from `models/` directory.")

# ==========================================
# TAB 3: INTERACTIVE PREDICTION ENGINE
# ==========================================
with tab3:
    st.header("🔮 Real-Time & Batch Prediction Engine")
    
    st.subheader("Option A: Batch CSV Prediction Upload")
    uploaded_file = st.file_uploader("Upload CSV file for inference", type=["csv"])
    selected_model_name = st.selectbox("Select Model Architecture", list(models.keys()))
    
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Preview:", user_df.head(3))
        
        if st.button("🚀 Run Batch Prediction"):
            if preprocessor is not None and scaler is not None:
                target_col = 'deposit' if 'deposit' in user_df.columns else None
                X_pred_raw = user_df.drop(columns=[target_col]) if target_col in user_df.columns else user_df
                
                try:
                    X_pred_proc = preprocessor.transform(X_pred_raw)
                    X_pred_scaled = scaler.transform(X_pred_proc)
                    
                    model = models[selected_model_name]
                    predictions = model.predict(X_pred_scaled)
                    
                    user_df['Prediction'] = ['Subscribed (Yes)' if p == 1 else 'Not Subscribed (No)' for p in predictions]
                    
                    st.success("Batch predictions completed successfully!")
                    st.dataframe(user_df, use_container_width=True)
                    
                    csv = user_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name='bank_predictions.csv',
                        mime='text/csv'
                    )
                except Exception as e:
                    st.error(f"Prediction execution failed: {e}")
