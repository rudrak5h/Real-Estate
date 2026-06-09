# 🏠 Real Estate Investment Advisor

## 📌 Project Overview

The Real Estate Investment Advisor is a Machine Learning project that predicts whether a property is a good investment and estimates its future price after 5 years.

The project uses:

* Classification models for investment prediction
* Regression models for future price prediction
* Feature engineering
* SMOTE balancing
* MLflow experiment tracking
* Streamlit deployment

---

# 🚀 Features

* Predict Good or Bad Investment
* Predict Future Property Price
* Interactive Streamlit Web App
* Data Visualization Dashboard
* Feature Importance Analysis
* MLflow Model Tracking
* SMOTE for Class Balancing

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Streamlit
* MLflow
* Imbalanced-learn

---

# 📂 Project Structure

```text
Real-Estate-Investment-Advisor/
│
├── app.py
├── model_training.ipynb
├── data_pre.ipynb
├── cleaned.csv
├── requirements.txt
├── README.md
│
├── best_classification_model.pkl
├── best_regression_model.pkl
├── scaler.pkl
├── label_encoders.pkl
├── feature_columns.pkl
```

---

# ⚙️ Machine Learning Workflow

## 1. Data Preprocessing

* Handling missing values
* Encoding categorical features
* Feature scaling

## 2. Feature Engineering

Created new features:

* Price_per_SqFt
* Transport_Score
* Amenities_Count
* Price_Value_Score

## 3. Target Engineering

Created custom target variable:

* Good_Investment

## 4. Handling Imbalanced Data

Used SMOTE to balance class distribution.

## 5. Model Training

### Classification Models

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

### Regression Models

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

---

# 📊 Evaluation Metrics

## Classification

* Accuracy
* Precision
* Recall
* F1 Score

## Regression

* MAE
* RMSE
* R² Score

---

# 📈 MLflow Tracking

MLflow was used for:

* Experiment tracking
* Metric logging
* Parameter logging
* Model logging

---

# 🌐 Streamlit Application

Run the app using:

```bash
streamlit run app.py
```

---

# 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 👨‍💻 Author

Rudraksh Bhardwaj
