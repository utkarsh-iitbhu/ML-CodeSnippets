# Machine Learning:
## List of topics covered in this repo
_____________________________________________________________________________________

## 1. Data Loading
- CSV Load file:
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

- API to Dataframe
- Working with Json
- Working with SQL
_____________________________________________________________________________________

## 2. EDA 

**1. Import and Explore:**
- a. Imports
    - Import packages
    - Import data in DataFrame

- b. Explore
    - Generate ProfileReport
    - Top 5 records
    - Shape
    - Summary 
    - Datatype
    - Duplicate
    - Missing
    - Correlation

**2. EDA:**
- a. Convert Data columns
    - Ordinal
    - Nominal
    - Discrete
    - Continuos

- b. Analysis
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

**1. Feature Transformation:**

- a. Missing Value Imputation
    - Simple Imputation
    - Mean/Median/Mode Imputation
    - Forward/Backward Fill
    - Interpolation Methods (Linear, Polynomial, Spline)
    - K-Nearest Neighbors (KNN) Imputation
    - Predictive Modeling (e.g., Regression Models)

- b. Handling Categorical Features
    - One-Hot Encoding
    - Label Encoding
    - Ordinal Encoding
    - Binary Encoding
    - Target Encoding
    - Frequency Encoding

- c. Outlier Detection
    - Z-Score
    - IQR (Interquartile Range)
    - Modified Z-Score
    - DBScan Clustering
    - Isolation Forest
    - Local Outlier Factor(LOF)

- d. Feature Scaling
    - Standard Scaler(Z-score Normalized)
    - MinMax Scaler
    - Max Abs Scaler
    - Robust Scaler

- e. Data Transformation
    - Log Transform
    - Box-Cox Transform
    - Yeo-Johnson Transform
    - Quantile Transform

**2. Feature Construction:**

- Date Features (Year, Month, Day, Hour, Minute, Second)
- Time Series Decomposition (Trend, Seasonality, Residuals)
- Rolling Window Statistics (Mean, Median, Std Dev)
- Lag Features
- Aggregation Features
- Domain Specific Features

**3. Feature Selection:**

- a. Filter Methods
    - Correlation-based Selection
    - Chi-Square Test
    - ANOVA F-Test
    - Mutual Information

- b. Wrapper Methods
    - Recursive Feature Elimination (RFE)
    - Forward/Backward Selection

- c. Embedded Methods
    - Lasso Regularization
    - Ridge Regularization
    - Random Forest Importance

- d. Dimensionality Reduction Techniques
    - Principal Component Analysis (PCA)
    - Linear Discriminant Analysis (LDA)
    - t-SNE

**4. Feature Extraction:**

- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- Kernel PCA
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Uniform Manifold Approximation and Projection (UMAP)
-  Independent Component Analysis (ICA)
- Non-negative Matrix Factorization (NMF)
- Singular Value Decomposition (SVD)

_____________________________________________________________________________________

## 4. Models

**1. Model Selection and Training**
   - 1.1 Classification Models
      - Logistic Regression
      - Decision Trees
      - Random Forest
      - Support Vector Machines (SVM)
      - Gradient Boosting (e.g., XGBoost, LightGBM)
      - K-Nearest Neighbors (KNN)
      - Naive Bayes
      - Neural Networks

   - 1.2 Regression Models
      - Linear Regression
      - Polynomial Regression
      - Ridge Regression
      - Lasso Regression
      - Elastic Net
      - Decision Tree Regressor
      - Random Forest Regressor
      - Gradient Boosting Regressor
      - Support Vector Regression (SVR)

   - 1.3 Clustering Models
      - K-Means
      - Hierarchical Clustering
      - DBSCAN
      - Gaussian Mixture Models
      - Mean Shift
      - Spectral Clustering

   - 1.4 Cross-Validation Techniques
      - K-Fold Cross-Validation
      - Stratified K-Fold Cross-Validation
      - Leave-One-Out Cross-Validation
      - Time Series Cross-Validation

   - 1.5 Handling Class Imbalance
      - Oversampling (e.g., SMOTE)
      - Undersampling
      - Combination (SMOTEENN, SMOTETomek)
      - Class Weights
      - Ensemble Methods (e.g., BalancedRandomForestClassifier)

**2. Hyperparameter Tuning:**
   - 2.1 Grid Search
   - 2.2 Random Search
   - 2.3 Bayesian Optimization
   - 2.4 Genetic Algorithms
   - 2.5 Hyperband
   - 2.6 Optuna

**3. Model Evaluation:**
   - 3.1 Classification Metrics
      - Accuracy
      - Precision
      - Recall
      - F1-Score
      - ROC-AUC
      - PR-AUC

   - 3.2 Regression Metrics
      - Mean Squared Error (MSE)
      - Root Mean Squared Error (RMSE)
      - Mean Absolute Error (MAE)
      - R-squared (R2)
      - Adjusted R-squared

   - 3.3 Clustering Metrics
      - Silhouette Score
      - Calinski-Harabasz Index
      - Davies-Bouldin Index

   - 3.4 Reports and Visualizations
      - Confusion Matrix
      - Classification Report
      - ROC Curve
      - Precision-Recall Curve
      - Learning Curves
      - Validation Curves

**4. Model Interpretation:**
   - 4.1 Feature Importance Analysis
      - Random Forest Feature Importance
      - Permutation Importance
      - Recursive Feature Elimination (RFE)

   - 4.2 SHAP (SHapley Additive exPlanations)
      - SHAP Summary Plot
      - SHAP Dependence Plot
      - SHAP Force Plot
      - SHAP Interaction Values

   - 4.3 Partial Dependence Plots (PDP)
   - 4.4 Individual Conditional Expectation (ICE) Plots
   - 4.5 Global Surrogate Models
   - 4.6 Local Interpretable Model-agnostic Explanations (LIME)

**5. Ensemble Methods:**
   - 5.1 Bagging
      - Random Forest
      - Bagging Classifier/Regressor

   - 5.2 Boosting
      - AdaBoost
      - Gradient Boosting
      - XGBoost
      - LightGBM
      - CatBoost

   - 5.3 Stacking
      - StackingClassifier
      - StackingRegressor

   - 5.4 Voting
      - VotingClassifier
      - VotingRegressor

_____________________________________________________________________________________

## 5. Evaluation and Deployement

**6. Error Analysis:**
   - 6.1 Analyzing Misclassifications
      - Identifying patterns in misclassified instances
      - Confusion matrix analysis
      - Error rate by class

   - 6.2 Bias-Variance Tradeoff
      - Learning curves analysis
      - Bias-variance decomposition

   - 6.3 Residual Analysis (for regression)
      - Residual plots
      - Q-Q plots
      - Heteroscedasticity check

   - 6.4 Cross-validation Insights
      - K-fold CV score distribution
      - Out-of-fold predictions analysis

   - 6.5 Feature Importance in Errors
      - SHAP values for misclassifications
      - Feature importance for error cases

**7. Model Persistence:**
   - 7.1 Saving Models
      - Pickle serialization
      - Joblib serialization
      - TensorFlow SavedModel format
      - ONNX format

   - 7.2 Loading Models
      - Deserializing saved models
      - Versioning loaded models

   - 7.3 Model Versioning
      - Version control for models (e.g., DVC, MLflow)
      - Model metadata tracking

   - 7.4 Model Registry
      - Centralized model storage
      - Model lifecycle management

**8. Model Deployment:**
   - 8.1 API Development
      - Flask API
      - FastAPI
      - Django REST framework

   - 8.2 Containerization
      - Docker containerization
      - Docker-compose for multi-container apps

   - 8.3 Cloud Deployment
      - AWS SageMaker
      - Google Cloud AI Platform
      - Azure Machine Learning

   - 8.4 Serverless Deployment
      - AWS Lambda
      - Google Cloud Functions
      - Azure Functions

   - 8.5 Edge Deployment
      - TensorFlow Lite
      - ONNX Runtime

**9. Monitoring and Maintenance:**
   - 9.1 Logging
      - Application logging
      - Model prediction logging
      - Error logging

   - 9.2 Performance Monitoring
      - Model accuracy tracking
      - Prediction latency monitoring
      - Resource utilization monitoring

   - 9.3 Data Drift Detection
      - Feature distribution monitoring
      - Concept drift detection
      - Outlier detection in new data

   - 9.4 Automated Alerts
      - Performance degradation alerts
      - Data quality alerts
      - System health alerts

   - 9.5 Model Updating
      - Incremental learning
      - Periodic retraining
      - A/B testing for model updates

   - 9.6 Feedback Loop Implementation
      - User feedback collection
      - Ground truth acquisition
      - Continuous learning pipeline

**10. Advanced Techniques:**
   - 10.1 Automated Machine Learning (AutoML)
      - Auto-sklearn
      - TPOT
      - H2O AutoML
      - Google Cloud AutoML

____________________________________________________________________________
