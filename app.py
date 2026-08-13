import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing ML Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .sub-header {
        font-size: 1.05rem;
        text-align: center;
        color: #6B7280;
        margin-bottom: 25px;
    }

    .section-header {
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 10px;
    }

    .metric-card {
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-weight: 600;
        padding: 0 18px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-header">🏦 Bank Marketing Term Deposit Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-header">'
    'BITS Pilani M.Tech Data Science & Engineering | Machine Learning Assignment 2'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# LOAD ARTIFACTS
# ============================================================

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

            with open(path, "rb") as file:
                models[name] = pickle.load(file)

    preprocessor = None
    scaler = None

    if os.path.exists("models/preprocessor.pkl"):

        with open("models/preprocessor.pkl", "rb") as file:
            preprocessor = pickle.load(file)

    if os.path.exists("models/scaler.pkl"):

        with open("models/scaler.pkl", "rb") as file:
            scaler = pickle.load(file)

    return models, preprocessor, scaler


models, preprocessor, scaler = load_artifacts()

# ============================================================
# LOAD TEST DATA
# ============================================================

@st.cache_data
def load_test_data():

    path = "data/test_data.csv"

    if os.path.exists(path):
        return pd.read_csv(path)

    return None


test_df = load_test_data()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Dashboard Controls")

    st.markdown("---")

    st.subheader("📦 System Status")

    if test_df is not None:
        st.success("Test Dataset: Loaded")
    else:
        st.error("Test Dataset: Missing")

    if len(models) > 0:
        st.success(f"Models Loaded: {len(models)}")
    else:
        st.error("Models Missing")

    if preprocessor is not None:
        st.success("Preprocessor: Loaded")
    else:
        st.error("Preprocessor Missing")

    if scaler is not None:
        st.success("Scaler: Loaded")
    else:
        st.error("Scaler Missing")

    st.markdown("---")

    if test_df is not None:

        st.subheader("📊 Dataset Information")

        st.write(f"Rows: **{len(test_df):,}**")
        st.write(f"Columns: **{test_df.shape[1]:,}**")

        if "deposit" in test_df.columns:
            st.write("Target: **deposit**")

    st.markdown("---")

    st.caption(
        "Machine Learning Assignment 2\n\n"
        "BITS Pilani"
    )

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_binary_target(series):

    return series.apply(
        lambda x: 1
        if str(x).strip().lower() in ["yes", "1", "true"]
        else 0
    )


def subscription_rate(df, column):

    if column not in df.columns or "deposit" not in df.columns:
        return None

    result = (
        df.groupby(column)["deposit"]
        .apply(
            lambda x: (
                x.astype(str)
                .str.lower()
                .eq("yes")
                .mean()
                * 100
            )
        )
        .sort_values(ascending=False)
    )

    return result


def get_model_probability(model, X):

    if hasattr(model, "predict_proba"):

        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):

        scores = model.decision_function(X)

        scores = (scores - scores.min()) / (
            scores.max() - scores.min() + 1e-9
        )

        return scores

    return None


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 EDA & Data Explorer",
        "📈 Model Evaluation",
        "🔮 Prediction Engine",
        "📋 Test Dataset Explorer"
    ]
)

# ============================================================
# TAB 1 — EDA
# ============================================================

with tab1:

    st.header("📊 Exploratory Data Analysis & Visual Insights")

    if test_df is None:

        st.error("Test dataset not found at `data/test_data.csv`.")

    else:

        target_col = "deposit" if "deposit" in test_df.columns else None

        # ----------------------------------------------------
        # DATASET KPIs
        # ----------------------------------------------------

        st.subheader("📌 Dataset Overview")

        total_rows = len(test_df)

        feature_count = (
            test_df.shape[1] - 1
            if target_col
            else test_df.shape[1]
        )

        numeric_count = len(
            test_df.select_dtypes(
                include=np.number
            ).columns
        )

        categorical_count = (
            feature_count - numeric_count
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Records",
            f"{total_rows:,}"
        )

        c2.metric(
            "Features",
            feature_count
        )

        c3.metric(
            "Numerical Features",
            numeric_count
        )

        c4.metric(
            "Categorical Features",
            categorical_count
        )

        st.markdown("---")

        # ----------------------------------------------------
        # TARGET DISTRIBUTION
        # ----------------------------------------------------

        if target_col:

            st.subheader(
                "🎯 Target Variable — Deposit Subscription"
            )

            target_counts = test_df[target_col].value_counts()

            total = target_counts.sum()

            no_count = target_counts.get("no", 0)
            yes_count = target_counts.get("yes", 0)

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Total Customers",
                f"{total:,}"
            )

            c2.metric(
                "Subscribed",
                f"{yes_count:,}",
                f"{yes_count / total * 100:.2f}%"
            )

            c3.metric(
                "Not Subscribed",
                f"{no_count:,}",
                f"{no_count / total * 100:.2f}%"
            )

            fig, ax = plt.subplots(figsize=(7, 4))

            target_counts.plot(
                kind="bar",
                ax=ax
            )

            ax.set_title(
                "Deposit Subscription Count"
            )

            ax.set_xlabel("Deposit")
            ax.set_ylabel("Number of Customers")

            ax.tick_params(axis="x", rotation=0)

            st.pyplot(fig)

            st.info(
                "💡 The target distribution is calculated dynamically "
                "from the uploaded test dataset rather than being hard-coded."
            )

        st.markdown("---")

        # ----------------------------------------------------
        # DURATION ANALYSIS
        # ----------------------------------------------------

        if "duration" in test_df.columns and target_col:

            st.subheader(
                "📞 Contact Duration vs Subscription"
            )

            c1, c2 = st.columns(2)

            with c1:

                fig, ax = plt.subplots(figsize=(7, 4))

                sns.boxplot(
                    data=test_df,
                    x="deposit",
                    y="duration",
                    ax=ax
                )

                ax.set_title(
                    "Call Duration by Deposit Outcome"
                )

                ax.set_xlabel("Deposit")
                ax.set_ylabel("Duration (seconds)")

                st.pyplot(fig)

            with c2:

                duration_summary = (
                    test_df
                    .groupby("deposit")["duration"]
                    .agg(["mean", "median"])
                    .reset_index()
                )

                fig, ax = plt.subplots(figsize=(7, 4))

                sns.barplot(
                    data=duration_summary,
                    x="deposit",
                    y="mean",
                    ax=ax
                )

                ax.set_title(
                    "Average Call Duration"
                )

                ax.set_xlabel("Deposit")
                ax.set_ylabel(
                    "Average Duration (seconds)"
                )

                st.pyplot(fig)

            st.info(
                "💡 Longer call durations are generally associated "
                "with higher subscription rates in this dataset."
            )

        st.markdown("---")

        # ----------------------------------------------------
        # CATEGORICAL FEATURE ANALYSIS
        # ----------------------------------------------------

        st.subheader(
            "👥 Customer Segment Analysis"
        )

        categorical_features = [
            "job",
            "marital",
            "education",
            "housing",
            "loan",
            "contact",
            "poutcome"
        ]

        available_categorical = [
            c
            for c in categorical_features
            if c in test_df.columns
        ]

        if available_categorical:

            selected_category = st.selectbox(
                "Select a categorical feature",
                available_categorical
            )

            rate = subscription_rate(
                test_df,
                selected_category
            )

            if rate is not None:

                fig, ax = plt.subplots(
                    figsize=(10, 5)
                )

                rate.plot(
                    kind="bar",
                    ax=ax
                )

                ax.set_title(
                    f"Subscription Rate by {selected_category.title()}"
                )

                ax.set_xlabel(
                    selected_category.title()
                )

                ax.set_ylabel(
                    "Subscription Rate (%)"
                )

                ax.tick_params(
                    axis="x",
                    rotation=45
                )

                st.pyplot(fig)

                st.info(
                    "💡 This chart shows the percentage of customers "
                    "within each category who subscribed to a term deposit."
                )

        st.markdown("---")

        # ----------------------------------------------------
        # AGE DISTRIBUTION
        # ----------------------------------------------------

        if "age" in test_df.columns:

            st.subheader(
                "🎂 Age Distribution"
            )

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            if target_col:

                sns.histplot(
                    data=test_df,
                    x="age",
                    hue="deposit",
                    kde=True,
                    bins=25,
                    ax=ax
                )

            else:

                sns.histplot(
                    data=test_df,
                    x="age",
                    kde=True,
                    bins=25,
                    ax=ax
                )

            ax.set_title(
                "Customer Age Distribution"
            )

            ax.set_xlabel("Age")
            ax.set_ylabel("Customer Count")

            st.pyplot(fig)

        st.markdown("---")

        # ----------------------------------------------------
        # NUMERICAL FEATURE ANALYSIS
        # ----------------------------------------------------

        st.subheader(
            "📐 Numerical Feature Explorer"
        )

        numerical_features = test_df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numerical_features:

            selected_numeric = st.selectbox(
                "Select numerical feature",
                numerical_features
            )

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            sns.histplot(
                data=test_df,
                x=selected_numeric,
                kde=True,
                bins=30,
                ax=ax
            )

            ax.set_title(
                f"Distribution of {selected_numeric}"
            )

            st.pyplot(fig)

        st.markdown("---")

        # ----------------------------------------------------
        # CORRELATION MATRIX
        # ----------------------------------------------------

        st.subheader(
            "🔥 Numerical Feature Correlation Matrix"
        )

        if len(numerical_features) >= 2:

            corr = test_df[
                numerical_features
            ].corr()

            fig, ax = plt.subplots(
                figsize=(10, 7)
            )

            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                linewidths=0.5,
                ax=ax
            )

            ax.set_title(
                "Correlation Matrix"
            )

            st.pyplot(fig)

            st.info(
                "💡 Correlation analysis helps identify strongly "
                "related numerical features and potential multicollinearity."
            )


# ============================================================
# TAB 2 — MODEL EVALUATION
# ============================================================

with tab2:

    st.header(
        "📈 Model Performance & Evaluation"
    )

    if (
        test_df is not None
        and len(models) > 0
        and preprocessor is not None
        and scaler is not None
    ):

        target_col = (
            "deposit"
            if "deposit" in test_df.columns
            else test_df.columns[-1]
        )

        X_test_raw = test_df.drop(
            columns=[target_col]
        )

        y_test = get_binary_target(
            test_df[target_col]
        )

        # Transform test data

        try:

            X_test_proc = preprocessor.transform(
                X_test_raw
            )

            X_test_scaled = scaler.transform(
                X_test_proc
            )

        except Exception as e:

            st.error(
                f"Preprocessing failed: {e}"
            )

            st.stop()

        eval_results = []

        model_cm = {}
        model_probs = {}
        model_predictions = {}

        # ----------------------------------------------------
        # MODEL EVALUATION
        # ----------------------------------------------------

        for name, model in models.items():

            y_pred = model.predict(
                X_test_scaled
            )

            y_prob = get_model_probability(
                model,
                X_test_scaled
            )

            eval_results.append(
                {
                    "Model": name,
                    "Accuracy": accuracy_score(
                        y_test,
                        y_pred
                    ),
                    "ROC-AUC": roc_auc_score(
                        y_test,
                        y_prob
                    ),
                    "Precision": precision_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    ),
                    "Recall": recall_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    ),
                    "F1 Score": f1_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    ),
                    "MCC Score": matthews_corrcoef(
                        y_test,
                        y_pred
                    )
                }
            )

            model_cm[name] = confusion_matrix(
                y_test,
                y_pred
            )

            model_probs[name] = y_prob
            model_predictions[name] = y_pred

        metrics_df = pd.DataFrame(
            eval_results
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        best_row = metrics_df.sort_values(
            "MCC Score",
            ascending=False
        ).iloc[0]

        best_model_name = best_row["Model"]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "🏆 Best Model",
            best_model_name
        )

        c2.metric(
            "Accuracy",
            f"{best_row['Accuracy']:.4f}"
        )

        c3.metric(
            "ROC-AUC",
            f"{best_row['ROC-AUC']:.4f}"
        )

        c4.metric(
            "MCC",
            f"{best_row['MCC Score']:.4f}"
        )

        st.success(
            f"🏆 **{best_model_name}** is the top-performing "
            f"model based on MCC."
        )

        st.markdown("---")

        # ----------------------------------------------------
        # PERFORMANCE TABLE
        # ----------------------------------------------------

        st.subheader(
            "📋 Model Performance Comparison"
        )

        st.dataframe(
            metrics_df.style.highlight_max(
                axis=0
            ),
            use_container_width=True
        )

        st.markdown("---")

        # ----------------------------------------------------
        # METRIC COMPARISON CHART
        # ----------------------------------------------------

        st.subheader(
            "📊 Model Metric Comparison"
        )

        chart_df = metrics_df.set_index(
            "Model"
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        chart_df[
            [
                "Accuracy",
                "ROC-AUC",
                "Precision",
                "Recall",
                "F1 Score",
                "MCC Score"
            ]
        ].plot(
            kind="bar",
            ax=ax
        )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title(
            "Comparison of Classification Metrics"
        )

        plt.xticks(rotation=30)

        st.pyplot(fig)

        # ----------------------------------------------------
        # ROC CURVES
        # ----------------------------------------------------

        st.subheader(
            "📈 ROC Curve Comparison"
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        for name in models.keys():

            fpr, tpr, _ = roc_curve(
                y_test,
                model_probs[name]
            )

            auc_value = roc_auc_score(
                y_test,
                model_probs[name]
            )

            ax.plot(
                fpr,
                tpr,
                label=f"{name} (AUC={auc_value:.3f})"
            )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        ax.set_xlabel(
            "False Positive Rate"
        )

        ax.set_ylabel(
            "True Positive Rate"
        )

        ax.set_title(
            "ROC Curve — All Models"
        )

        ax.legend()

        st.pyplot(fig)

        # ----------------------------------------------------
        # PRECISION-RECALL CURVES
        # ----------------------------------------------------

        st.subheader(
            "🎯 Precision-Recall Curve Comparison"
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        for name in models.keys():

            precision, recall, _ = precision_recall_curve(
                y_test,
                model_probs[name]
            )

            ax.plot(
                recall,
                precision,
                label=name
            )

        ax.set_xlabel(
            "Recall"
        )

        ax.set_ylabel(
            "Precision"
        )

        ax.set_title(
            "Precision-Recall Curves"
        )

        ax.legend()

        st.pyplot(fig)

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        st.subheader(
            "🔍 Confusion Matrix"
        )

        selected_model = st.selectbox(
            "Select Model",
            list(models.keys()),
            key="cm_model"
        )

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        sns.heatmap(
            model_cm[selected_model],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "No",
                "Yes"
            ],
            yticklabels=[
                "No",
                "Yes"
            ],
            ax=ax
        )

        ax.set_xlabel(
            "Predicted Label"
        )

        ax.set_ylabel(
            "Actual Label"
        )

        ax.set_title(
            f"Confusion Matrix — {selected_model}"
        )

        st.pyplot(fig)

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "🌳 Random Forest Feature Importance"
        )

        rf_model = models.get(
            "Random Forest"
        )

        if rf_model is not None and hasattr(
            rf_model,
            "feature_importances_"
        ):

            importances = (
                rf_model
                .feature_importances_
            )

            try:

                feature_names = (
                    preprocessor
                    .get_feature_names_out()
                )

            except Exception:

                feature_names = [
                    f"Feature {i+1}"
                    for i in range(
                        len(importances)
                    )
                ]

            importance_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": importances
                }
            ).sort_values(
                "Importance",
                ascending=False
            )

            top_n = st.slider(
                "Number of features to display",
                min_value=5,
                max_value=min(
                    20,
                    len(importance_df)
                ),
                value=min(
                    10,
                    len(importance_df)
                )
            )

            top_features = importance_df.head(
                top_n
            )

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            sns.barplot(
                data=top_features,
                x="Importance",
                y="Feature",
                ax=ax
            )

            ax.set_title(
                "Top Random Forest Features"
            )

            st.pyplot(fig)

            st.dataframe(
                importance_df,
                use_container_width=True
            )

        else:

            st.info(
                "Random Forest feature importance is unavailable."
            )

    else:

        st.warning(
            "⚠️ Required models, scaler, preprocessor, "
            "or test dataset are missing."
        )


# ============================================================
# TAB 3 — PREDICTION ENGINE
# ============================================================

with tab3:

    st.header(
        "🔮 Real-Time & Batch Prediction Engine"
    )

    if len(models) == 0:

        st.error(
            "No trained models were found."
        )

    else:

        selected_model_name = st.selectbox(
            "🤖 Select Model",
            list(models.keys()),
            key="prediction_model"
        )

        prediction_mode = st.radio(
            "Select Prediction Mode",
            [
                "Single Customer",
                "Batch CSV"
            ],
            horizontal=True
        )

        # ====================================================
        # SINGLE CUSTOMER
        # ====================================================

        if prediction_mode == "Single Customer":

            st.subheader(
                "👤 Single Customer Prediction"
            )

            if test_df is None:

                st.warning(
                    "Test dataset is required to generate "
                    "the prediction form."
                )

            else:

                target_col = (
                    "deposit"
                    if "deposit" in test_df.columns
                    else None
                )

                feature_df = (
                    test_df.drop(
                        columns=[target_col]
                    )
                    if target_col
                    else test_df.copy()
                )

                input_data = {}

                columns = feature_df.columns

                col1, col2 = st.columns(2)

                for i, column in enumerate(
                    columns
                ):

                    series = feature_df[column]

                    with (
                        col1
                        if i % 2 == 0
                        else col2
                    ):

                        if (
                            pd.api.types
                            .is_numeric_dtype(series)
                        ):

                            default_value = float(
                                series.median()
                            )

                            input_data[column] = st.number_input(
                                column,
                                value=default_value
                            )

                        else:

                            options = (
                                series
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                            )

                            input_data[column] = st.selectbox(
                                column,
                                options
                            )

                st.markdown("---")

                if st.button(
                    "🚀 Predict Subscription",
                    use_container_width=True
                ):

                    input_df = pd.DataFrame(
                        [input_data]
                    )

                    try:

                        X_proc = (
                            preprocessor
                            .transform(input_df)
                        )

                        X_scaled = (
                            scaler
                            .transform(X_proc)
                        )

                        model = models[
                            selected_model_name
                        ]

                        prediction = model.predict(
                            X_scaled
                        )[0]

                        probability = (
                            get_model_probability(
                                model,
                                X_scaled
                            )[0]
                        )

                        if prediction == 1:

                            st.success(
                                "🎉 Customer is predicted to SUBSCRIBE."
                            )

                        else:

                            st.warning(
                                "Customer is predicted NOT TO SUBSCRIBE."
                            )

                        c1, c2 = st.columns(2)

                        c1.metric(
                            "Prediction",
                            "YES"
                            if prediction == 1
                            else "NO"
                        )

                        c2.metric(
                            "Subscription Probability",
                            f"{probability * 100:.2f}%"
                        )

                    except Exception as e:

                        st.error(
                            f"Prediction failed: {e}"
                        )

        # ====================================================
        # BATCH PREDICTION
        # ====================================================

        else:

            st.subheader(
                "📁 Batch CSV Prediction"
            )

            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=["csv"]
            )

            if uploaded_file is not None:

                user_df = pd.read_csv(
                    uploaded_file
                )

                st.write(
                    "### Uploaded Data Preview"
                )

                st.dataframe(
                    user_df.head(10),
                    use_container_width=True
                )

                if st.button(
                    "🚀 Run Batch Prediction",
                    use_container_width=True
                ):

                    try:

                        target_col = (
                            "deposit"
                            if "deposit"
                            in user_df.columns
                            else None
                        )

                        X_pred_raw = (
                            user_df.drop(
                                columns=[target_col]
                            )
                            if target_col
                            else user_df.copy()
                        )

                        X_proc = (
                            preprocessor
                            .transform(
                                X_pred_raw
                            )
                        )

                        X_scaled = (
                            scaler
                            .transform(
                                X_proc
                            )
                        )

                        model = models[
                            selected_model_name
                        ]

                        predictions = (
                            model.predict(
                                X_scaled
                            )
                        )

                        probabilities = (
                            get_model_probability(
                                model,
                                X_scaled
                            )
                        )

                        result_df = user_df.copy()

                        result_df[
                            "Prediction"
                        ] = [
                            "Subscribed (Yes)"
                            if p == 1
                            else "Not Subscribed (No)"
                            for p in predictions
                        ]

                        result_df[
                            "Probability"
                        ] = (
                            probabilities * 100
                        ).round(2)

                        st.success(
                            "✅ Batch prediction completed successfully."
                        )

                        # ------------------------------------
                        # Prediction KPIs
                        # ------------------------------------

                        yes_predictions = (
                            predictions == 1
                        ).sum()

                        no_predictions = (
                            predictions == 0
                        ).sum()

                        c1, c2, c3 = st.columns(3)

                        c1.metric(
                            "Total Records",
                            len(result_df)
                        )

                        c2.metric(
                            "Predicted Yes",
                            yes_predictions
                        )

                        c3.metric(
                            "Predicted No",
                            no_predictions
                        )

                        st.markdown("---")

                        st.subheader(
                            "📊 Prediction Distribution"
                        )

                        prediction_counts = (
                            result_df[
                                "Prediction"
                            ].value_counts()
                        )

                        fig, ax = plt.subplots(
                            figsize=(8, 4)
                        )

                        prediction_counts.plot(
                            kind="bar",
                            ax=ax
                        )

                        ax.set_title(
                            "Batch Prediction Distribution"
                        )

                        ax.set_ylabel(
                            "Number of Customers"
                        )

                        ax.tick_params(
                            axis="x",
                            rotation=0
                        )

                        st.pyplot(fig)

                        st.markdown("---")

                        st.subheader(
                            "📋 Prediction Results"
                        )

                        st.dataframe(
                            result_df,
                            use_container_width=True
                        )

                        csv = (
                            result_df
                            .to_csv(
                                index=False
                            )
                            .encode("utf-8")
                        )

                        st.download_button(
                            label="📥 Download Prediction Results",
                            data=csv,
                            file_name=(
                                "bank_predictions.csv"
                            ),
                            mime="text/csv",
                            use_container_width=True
                        )

                    except Exception as e:

                        st.error(
                            f"Prediction execution failed: {e}"
                        )


# ============================================================
# TAB 4 — TEST DATASET EXPLORER
# ============================================================

with tab4:

    st.header(
        "📋 Test Dataset Explorer"
    )

    if test_df is None:

        st.warning(
            "Test dataset not found."
        )

    else:

        st.subheader(
            "🔎 Explore Test Records"
        )

        # ----------------------------------------------------
        # FILTER
        # ----------------------------------------------------

        filter_columns = st.multiselect(
            "Select columns to filter",
            test_df.columns.tolist()
        )

        filtered_df = test_df.copy()

        for column in filter_columns:

            unique_values = (
                test_df[column]
                .dropna()
                .unique()
                .tolist()
            )

            if len(unique_values) <= 30:

                selected_values = st.multiselect(
                    f"Filter {column}",
                    unique_values,
                    default=unique_values
                )

                filtered_df = filtered_df[
                    filtered_df[column]
                    .isin(selected_values)
                ]

        # ----------------------------------------------------
        # ROW COUNT
        # ----------------------------------------------------

        st.info(
            f"Showing **{len(filtered_df):,}** "
            f"of **{len(test_df):,}** records."
        )

        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=500
        )

        # ----------------------------------------------------
        # SUMMARY STATISTICS
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "📐 Statistical Summary"
        )

        st.dataframe(
            filtered_df.describe(
                include="all"
            ).T,
            use_container_width=True
        )

        # ----------------------------------------------------
        # MISSING VALUES
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "🧹 Missing Value Analysis"
        )

        missing_df = pd.DataFrame(
            {
                "Column": test_df.columns,
                "Missing Values": [
                    test_df[c].isna().sum()
                    for c in test_df.columns
                ]
            }
        )

        missing_df["Missing %"] = (
            missing_df["Missing Values"]
            / len(test_df)
            * 100
        )

        missing_df = missing_df.sort_values(
            "Missing Values",
            ascending=False
        )

        st.dataframe(
            missing_df,
            use_container_width=True
        )

        # ----------------------------------------------------
        # DOWNLOAD TEST DATA
        # ----------------------------------------------------

        csv = (
            filtered_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "📥 Download Filtered Test Data",
            data=csv,
            file_name="filtered_test_data.csv",
            mime="text/csv"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🏦 Bank Marketing Term Deposit Classification | "
    "BITS Pilani M.Tech Data Science & Engineering | "
    "Machine Learning Assignment 2"
)