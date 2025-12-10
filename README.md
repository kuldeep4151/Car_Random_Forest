# 🚗 Machine Learning Projects — Random Forest Models

This repository contains **two complete Machine Learning projects** using the **Random Forest** algorithm:

1. **Car Price Prediction using Random Forest Regression**  
2. **Digit Classification using Random Forest Classifier**

Both projects demonstrate data preprocessing, model building, evaluation, and prediction.

---

# 📌 1️⃣ Car Price Prediction using Random Forest

This project predicts **car resale price** based on attributes such as make, model, engine size, fuel type, mileage, and more.

---

## ✨ Project Overview

Car price prediction helps dealerships, buyers, and sellers understand market value.  
This project:

- Loads and cleans car price data  
- Encodes categorical features  
- Prepares numerical data  
- Trains a Random Forest Regression model  
- Evaluates performance  
- Predicts car prices  

---

## 🧹 Dataset Information

Typical dataset columns:

- Make  
- Model  
- Year  
- Transmission  
- FuelType  
- EngineSize  
- Mileage  
- Price (Target)

Preprocessing includes:

- Removing missing values  
- Label encoding  
- Train-test splitting  

---

## 🌲 Random Forest Regression

Random Forest is an ensemble of multiple decision trees.

### ✔ Benefits:
- Handles complex non-linear data  
- Provides feature importance  
- High accuracy  
- Robust to overfitting  

### 🔧 Key Hyperparameters:
- `n_estimators`  
- `max_depth`  
- `min_samples_split`  
- `min_samples_leaf`  

---

## 📊 Evaluation

- R² Score  
- MAE  
- MSE  
- RMSE  
- Feature Importance  

---

## 🔧 Improvements

- Hyperparameter tuning (GridSearchCV)  
- Try XGBoost / CatBoost  
- API deployment using Flask / FastAPI  
- Add additional car attributes  

---

# 2️⃣ 🧮 Digit Classification using Random Forest

This project classifies handwritten digits (0–9) using the **Digits dataset** from Scikit-Learn.

---

## ✨ Project Overview

The Random Forest Classifier is trained on **8×8 grayscale images** of digits.  
Each image has 64 features (pixel intensities).

The project demonstrates:

- Loading the digits dataset  
- Flattening image data  
- Training a Random Forest model  
- Evaluating performance  
- Predicting digit class for new input  

---

## 🌲 Random Forest Classifier

### ✔ Benefits:
- High accuracy  
- Handles noisy data well  
- Fast inference  
- No need for feature scaling  

---

## 📊 Evaluation Metrics (Digits)

- Classification Accuracy  
- Confusion Matrix  
- Per-class Precision & Recall  

Example accuracy:

Accuracy: ~96–98%

---

## 🔧 Possible Improvements

- Try SVM or KNN for comparison  
- Add cross-validation  
- Visualize misclassifications  
- Deploy as a small prediction app  

---

# 🛠 Technologies Used

- Python  
- NumPy  
- Pandas  
- Scikit-Learn  
- Matplotlib  
- Seaborn  

---

# 🙌 Author

**Kuldeep Patel**  
Machine Learning & Data Science Practitioner  

If you like this project, please ⭐ the repository!




