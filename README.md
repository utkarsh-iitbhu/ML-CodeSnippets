
## 1. Data Loading
**a. CSV Load file:**
- Local csv
- csv from url
- sep param
- index_col
- header param
- use_cols
- skiprows/nrows
- Encoding params
- dtypes params
- handling dates
- huge data in chunks

**b. API to Dataframe**
**c. Working with Json**
**d. Working with SQL**
_____________________________________________________________________________________

## 2. EDA 

**a. Import and Explore:**
- i. Imports
    - Import packages
    - Import data in DataFrame

- ii. Explore
    - Generate ProfileReport
    - Top 5 records
    - Shape
    - Summary 
    - Datatype
    - Duplicate
    - Missing
    - Correlation

**b. EDA:**
- i. Convert Data columns
    - Ordinal
    - Nominal
    - Discrete
    - Continuous

- ii. Analysis
    - Univariate
        - Numerical
        - Categorical
    - Bivariate
        - Numerical vs Numerical
        - Numerical vs Categorical
        - Categorical vs Categorical
    - Multivariate
    - QQ Plots
    
_____________________________________________________________________________________

## 3. Feature Engineering

**a. Feature Transformation:**

- i. Missing Value Imputation
    - Simple Imputation
    - Mean/Median/Mode Imputation
    - Forward/Backward Fill
    - Interpolation Methods (Linear, Polynomial, Spline)
    - K-Nearest Neighbors (KNN) Imputation
    - Predictive Modeling (e.g., Regression Models)

- ii. Handling Categorical Features
    - One-Hot Encoding
    - Label Encoding
    - Ordinal Encoding
    - Binary Encoding
    - Target Encoding
    - Frequency Encoding

- iii. Outlier Detection
    - Z-Score
    - IQR (Interquartile Range)
    - Modified Z-Score
    - DBScan Clustering
    - Isolation Forest
    - Local Outlier Factor (LOF)

- iv. Feature Scaling
    - Standard Scaler (Z-score Normalized)
    - MinMax Scaler
    - Max Abs Scaler
    - Robust Scaler

- v. Data Transformation
    - Log Transform
    - Box-Cox Transform
    - Yeo-Johnson Transform
    - Quantile Transform

**b. Feature Construction:**

- i. Date Features (Year, Month, Day, Hour, Minute, Second)
- ii. Time Series Decomposition (Trend, Seasonality, Residuals)
- iii. Rolling Window Statistics (Mean, Median, Std Dev)
- iv. Lag Features
- v. Aggregation Features
- vi. Domain Specific Features

**c. Feature Selection:**

- i. Filter Methods
    - Correlation-based Selection
    - Chi-Square Test
    - ANOVA F-Test
    - Mutual Information

- ii. Wrapper Methods
    - Recursive Feature Elimination (RFE)
    - Forward/Backward Selection

- iii. Embedded Methods
    - Lasso Regularization
    - Ridge Regularization
    - Random Forest Importance

- iv. Dimensionality Reduction Techniques
    - Principal Component Analysis (PCA)
    - Linear Discriminant Analysis (LDA)
    - t-SNE

**d. Feature Extraction:**

- i. Principal Component Analysis (PCA)
- ii. Linear Discriminant Analysis (LDA)
- iii. Kernel PCA
- iv. t-Distributed Stochastic Neighbor Embedding (t-SNE)
- v. Uniform Manifold Approximation and Projection (UMAP)
- vi. Independent Component Analysis (ICA)
- vii. Non-negative Matrix Factorization (NMF)
- viii. Singular Value Decomposition (SVD)

_____________________________________________________________________________________

## 4. Models

**a. Model Selection and Training**

- i. Classification Models
    - Logistic Regression
    - Decision Trees
    - Random Forest
    - Support Vector Machines (SVM)
    - Gradient Boosting (e.g., XGBoost, LightGBM)
    - K-Nearest Neighbors (KNN)
    - Naive Bayes
    - Neural Networks

- ii. Regression Models
    - Linear Regression
    - Polynomial Regression
    - Ridge Regression
    - Lasso Regression
    - Elastic Net
    - Decision Tree Regressor
    - Random Forest Regressor
    - Gradient Boosting Regressor
    - Support Vector Regression (SVR)

- iii. Clustering Models
    - K-Means
    - Hierarchical Clustering
    - DBSCAN
    - Gaussian Mixture Models
    - Mean Shift
    - Spectral Clustering

- iv. Cross-Validation Techniques
    - K-Fold Cross-Validation
    - Stratified K-Fold Cross-Validation
    - Leave-One-Out Cross-Validation
    - Time Series Cross-Validation

- v. Handling Class Imbalance
    - Oversampling (e.g., SMOTE)
    - Undersampling
    - Combination (SMOTEENN, SMOTETomek)
    - Class Weights
    - Ensemble Methods (e.g., BalancedRandomForestClassifier)

**b. Hyperparameter Tuning:**
- i. Grid Search
- ii. Random Search
- iii. Bayesian Optimization
- iv. Genetic Algorithms
- v. Hyperband
- vi. Optuna

**c. Model Evaluation:**
- i. Classification Metrics
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC
    - PR-AUC

- ii. Regression Metrics
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - R-squared (R2)
    - Adjusted R-squared

- iii. Clustering Metrics
    - Silhouette Score
    - Calinski-Harabasz Index
    - Davies-Bouldin Index

- iv. Reports and Visualizations
    - Confusion Matrix
    - Classification Report
    - ROC Curve
    - Precision-Recall Curve
    - Learning Curves
    - Validation Curves

**d. Model Interpretation:**
- i. Feature Importance Analysis
    - Random Forest Feature Importance
    - Permutation Importance
    - Recursive Feature Elimination (RFE)

- ii. SHAP (SHapley Additive exPlanations)
    - SHAP Summary Plot
    - SHAP Dependence Plot
    - SHAP Force Plot
    - SHAP Interaction Values

- iii. Partial Dependence Plots (PDP)
- iv. Individual Conditional Expectation (ICE) Plots
- v. Global Surrogate Models
- vi. Local Interpretable Model-agnostic Explanations (LIME)

**e. Ensemble Methods:**
- i. Bagging
    - Random Forest
    - Bagging Classifier/Regressor

- ii. Boosting
    - AdaBoost
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - CatBoost

- iii. Stacking
    - StackingClassifier
    - StackingRegressor

- iv. Voting
    - VotingClassifier
    - VotingRegressor

_____________________________________________________________________________________

## 5. Evaluation and Deployment

**a. Error Analysis:**
- i. Analyzing Misclassifications
    - Identifying patterns in misclassified instances
    - Confusion matrix analysis
    - Error rate by class

- ii. Bias-Variance Tradeoff
    - Learning curves analysis
    - Bias-variance decomposition

- iii. Residual Analysis (for regression)
    - Residual plots
    - Q-Q plots
    - Heteroscedasticity check

- iv. Cross-validation Insights
    - K-fold CV score distribution
    - Out-of-fold predictions analysis

- v. Feature Importance in Errors
    - SHAP values for misclassifications
    - Feature importance for error cases

**b. Model Persistence:**
- i. Saving Models
    - Pickle serialization
    - Joblib serialization
    - TensorFlow SavedModel format
    - ONNX format

- ii. Loading Models
    - Deserializing saved models
    - Versioning loaded models

- iii. Model Versioning
    - Version control for models (e.g., DVC, MLflow)
    - Model metadata tracking

- iv. Model Registry
    - Centralized model storage
    - Model lifecycle management

**c. Model Deployment:**
- i. API Development
    - Flask API
    - FastAPI
    - Django REST framework

- ii. Containerization
    - Docker containerization
    - Docker-compose for multi-container apps

- iii. Cloud Deployment
    - AWS SageMaker
    - Google Cloud AI Platform
    - Azure Machine Learning

- iv. Serverless Deployment
    - AWS Lambda
    - Google Cloud Functions
    - Azure Functions

- v. Edge Deployment
    - TensorFlow Lite
    - ONNX Runtime

**d. Monitoring and Maintenance:**
- i. Logging
    - Application logging
    - Model prediction logging
    - Error logging

- ii. Performance Monitoring
    - Model accuracy tracking
    - Prediction latency monitoring
    - Resource utilization monitoring

- iii. Data Drift Detection
    - Feature distribution monitoring
    - Concept drift detection
    - Outlier detection in new data

- iv. Automated Alerts:
    - Performance degradation alerts
    - Data quality alerts
    - System health alerts

- v. Model Updating:
    - Incremental learning
    - Periodic retraining
    - A/B testing for model updates

- vi. Feedback Loop Implementation:**
    - User feedback collection
    - Ground truth acquisition
    - Continuous learning pipeline

**10. Advanced Techniques:**
- i. Automated Machine Learning (AutoML)
    - Auto-sklearn
    - TPOT
    - H2O AutoML
    - Google Cloud AutoML
____________________________________________________________________________
