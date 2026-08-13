# 🏦 Bank Marketing Term Deposit Classification Pipeline
**Program:** M.Tech Data Science & Engineering | BITS Pilani  
**Course:** Machine Learning Assignment 2  

---

## 📌 Project Overview
This project delivers an end-to-end Machine Learning pipeline and interactive Streamlit web application that predicts whether a banking client will subscribe to a term deposit based on direct marketing campaign attributes. The system features dynamic data ingestion via `kagglehub`, defensive imputation, feature scaling, adaptive class balancing (SMOTE), multi-model benchmarking across 6 core evaluation metrics, and artifact serialization for real-time app inference.

---

## 📏 Feature Scaling Rationale (`StandardScaler`)
Feature scaling normalizes all numerical features to have a mean of $0$ and a standard deviation of $1$.

* **Why Scaling is Necessary:** 
  * **Distance & Gradient-Sensitive Models:** Algorithms such as K-Nearest Neighbors (KNN) calculate Euclidean distances between data points, while Logistic Regression optimizes cost functions using gradient descent. Unscaled features with large ranges (e.g., `balance` spanning $-6,847$ to $81,204$) would completely dominate attributes with smaller ranges (e.g., `campaign` calls ranging $1$ to $63$), distorting distance metrics and slowing gradient convergence.
  * **Tree-Based Models:** Decision Trees and Random Forests split nodes based on individual feature thresholds and are invariant to scale. However, global scaling is applied across the dataset so a single, standardized feature matrix can be fed reliably into all 5 model architectures.

---

## ⚖️ Conditional SMOTE Implementation
Synthetic Minority Over-sampling Technique (SMOTE) generates synthetic samples along line segments connecting k-nearest neighbors in the minority class.

* **Adaptive Execution Logic:** Instead of blindly oversampling every dataset, the pipeline inspects the training class distribution (`y_train`) after the train-test split:
  $$\text{Minority Ratio} = \frac{\text{Count}(\text{Minority Class})}{\text{Total Training Samples}}$$
* **Thresholding Decision:** 
  * If $\text{Minority Ratio} < 0.35$, SMOTE is triggered to rebalance the training partition.
  * If $\text{Minority Ratio} \ge 0.35$, SMOTE is bypassed to preserve the natural data distribution and prevent synthetic noise.
* **Dataset Context:** In the primary Bank Marketing dataset (`bank.csv`), the target distribution is relatively balanced (~$52.6\%$ `no` vs ~$47.4\%$ `yes`). Bypassing SMOTE avoids creating redundant synthetic points, while preserving the conditional check guarantees that the pipeline can safely handle heavily imbalanced datasets in production.
* **Preventing Data Leakage:** SMOTE is applied **strictly to the training split (`X_train`, `y_train`)** after scaling, leaving the test split (`X_test`, `y_test`) untouched to preserve authentic performance metrics.

---

## 🔍 Exploratory Data Analysis (EDA) & Key Inferences

1. **Target Class Balance:**
   * **Observation:** The dataset split stands at $52.6\%$ non-subscribers and $47.4\%$ subscribers.
   * **Inference:** The target variable exhibits healthy natural balance, enabling models to learn decision boundaries for both outcomes without extreme majority class bias.

2. **Contact Duration vs. Subscription Success:**
   * **Observation:** Clients who subscribed had significantly longer call durations (median duration is noticeably higher for `yes` outcomes).
   * **Inference:** Call duration (`duration`) is one of the strongest individual predictors of deposit subscription success.

3. **Demographic & Occupational Patterns:**
   * **Observation:** Management, Technicians, and Admin roles account for the highest total volume of conversions, while Students and Retirees demonstrate the highest relative conversion rates.
   * **Inference:** Targeted marketing strategies focusing on specific job categories can maximize resource efficiency and campaign ROI.

4. **Multicollinearity Inspection:**
   * **Observation:** A masked correlation heatmap revealed low pairwise correlation values across numerical variables ($r < 0.50$).
   * **Inference:** Minimal severe multicollinearity exists among independent variables, ensuring stable coefficient estimation for linear models like Logistic Regression.

---

## ⚙️ Modeling Methodology
The preprocessed feature matrix is evaluated across 5 diverse classification algorithms:

1. **Logistic Regression:** Linear baseline using $L_2$ regularization.
2. **Decision Tree Classifier:** Non-linear single-tree baseline (`random_state=42`).
3. **K-Nearest Neighbors (KNN):** Distance-based non-parametric classifier.
4. **Gaussian Naive Bayes:** Probabilistic classifier assuming feature independence.
5. **Random Forest Classifier:** Ensemble architecture consisting of 100 decision trees trained with bootstrap aggregation.

---

## 🏆 Model Performance Results & Best Model Selection

Each model was tested on an isolated $20\%$ stratified test partition using 6 evaluation metrics:

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7962 | 0.8729 | 0.7831 | 0.7883 | 0.7857 | 0.5915 |
| **Decision Tree** | 0.7676 | 0.7662 | 0.7624 | 0.7401 | 0.7511 | 0.5334 |
| **KNN** | 0.7913 | 0.8503 | 0.7863 | 0.7684 | 0.7772 | 0.5811 |
| **Naive Bayes** | 0.7497 | 0.8077 | 0.7014 | 0.8214 | 0.7566 | 0.5088 |
| **Random Forest (Best)** | **0.8495** | **0.9133** | **0.8195** | **0.8752** | **0.8464** | **0.7007** |

### Key Performance Inferences:
* **Winning Model — Random Forest:** **Random Forest** achieved the top score across all key metrics with an **Accuracy of 84.95%**, an **ROC-AUC of 0.9133**, and an **MCC Score of 0.7007**.
* **Why MCC Matters:** Matthews Correlation Coefficient (MCC) accounts for all four confusion matrix quadrants (TP, TN, FP, FN). A score of **0.7007** confirms high prediction reliability and strong overall classification capability.
* **Ensemble Superiority:** Random Forest significantly outperformed single decision trees (84.95% vs 76.76% accuracy), demonstrating the effectiveness of ensemble bagging in reducing variance and mitigating overfitting.

---

## 📁 Repository Structure
```text
ML-Assignment2/
│── app.py                    # Multi-tab Streamlit Application
│── requirements.txt           # Python environment dependencies
│── README.md                  # Comprehensive Project Documentation
│── .gitignore                 # System & environment exclusion rules
├── data/
│   └── test_data.csv          # Stratified test dataset for app evaluation
└── models/
    ├── preprocessor.pkl       # SimpleImputer & OrdinalEncoder pipeline artifact
    ├── scaler.pkl             # StandardScaler pipeline artifact
    ├── logistic.pkl           # Trained Logistic Regression model
    ├── decision_tree.pkl      # Trained Decision Tree model
    ├── knn.pkl                # Trained KNN model
    ├── naive_bayes.pkl        # Trained Naive Bayes model
    └── random_forest.pkl      # Trained Random Forest model

