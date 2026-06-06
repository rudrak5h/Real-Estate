# 🏠 Real Estate Investment Advisor

A Machine Learning based Real Estate Investment Advisor that predicts whether a property is a good investment and estimates future property prices using Classification and Regression models.

---

# 📌 Project Overview

This project helps users:

* Predict whether a property is a **Good Investment**
* Estimate **Future Property Price after 5 Years**
* Analyze real estate trends using visual insights
* Compare multiple Machine Learning models
* Track experiments using MLflow
* Use an interactive Streamlit web application

---

# 🚀 Features

## ✅ Classification

Predicts whether a property is:

* Good Investment
* Not Good Investment

## ✅ Regression

Predicts:

* Estimated Property Price after 5 Years

## ✅ Streamlit Dashboard

Interactive UI with:

* Property filters
* Investment prediction
* Confidence score
* Future price prediction
* Visual insights

## ✅ MLflow Integration

Tracks:

* Experiments
* Metrics
* Parameters
* Model performance

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Matplotlib
* Seaborn
* MLflow
* Joblib

---

# 📂 Project Structure

```bash
Real-Estate-Investment-Advisor/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data_pre.ipynb
├── model_training.ipynb
│
├── best_classification_model.pkl
├── best_regression_model.pkl
├── scaler.pkl
├── label_encoders.pkl
├── feature_columns.pkl
│
└── screenshots/
```

---

# 📊 Machine Learning Models Used

## Classification Models

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

## Regression Models

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

---

# ⚙️ Feature Engineering

Engineered features include:

* Investment_Score
* Price_per_SqFt
* Transport_Score
* Amenities_Count

These features improve prediction quality and model performance.

---

# 📈 MLflow Experiment Tracking

MLflow is used for:

* Logging metrics
* Tracking experiments
* Comparing models
* Saving model runs

Experiments:

* Real_Estate_Classification
* Real_Estate_Regression

---

# ▶️ How to Run the Project

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Real-Estate-Investment-Advisor.git
```

---

## 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Generate Dataset

The cleaned dataset (`cleaned.csv`) is not uploaded due to GitHub file size limitations.

To generate it:

1. Place the original dataset in the project folder
2. Run:

```bash
data_pre.ipynb
```

This will generate:

```bash
cleaned.csv
```

---

## 4️⃣ Train Models

Run:

```bash
model_training.ipynb
```

This generates:

* Trained models
* Encoders
* Scaler
* Feature columns

---

## 5️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 6️⃣ Run MLflow UI

```bash
mlflow ui
```

Open:

```bash
http://127.0.0.1:5000
```

---

# 📷 Screenshots

Add screenshots for:

* Streamlit Dashboard
* Prediction Results
* MLflow Experiments
* Feature Importance Graphs

---

# 📌 Results

* Realistic investment prediction
* Future property price estimation
* Interactive ML dashboard
* MLflow experiment tracking
* Multiple ML model comparison

---

# 👨‍💻 Author

Rudraksh

---

# ⭐ Future Improvements

* Deploy on Streamlit Cloud
* Add real-time property APIs
* Add map-based visualization
* Improve recommendation engine
* Deep Learning integration
