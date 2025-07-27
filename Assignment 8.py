# -------------------------------
# 1. Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# -------------------------------
# 2. Load Dataset
# -------------------------------
df = pd.read_csv("Training Dataset.csv")
print(df.head())
print(df.info())

# -------------------------------
# 3. Handle Missing Values
# -------------------------------
# Fill categorical with mode, numeric with median
cat_cols = df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=['int64','float64']).columns

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# -------------------------------
# 4. Encode Categorical Variables
# -------------------------------
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# 5. Feature Engineering
# -------------------------------
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.drop(['Loan_ID'], axis=1, inplace=True)

# -------------------------------
# 6. Split Features & Target
# -------------------------------
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 7. Scale Numerical Features
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 8. Train Logistic Regression
# -------------------------------
log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------------
# 9. Train Random Forest
# -------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# -------------------------------
# 10. Save Best Model
# -------------------------------
joblib.dump(rf_model, "loan_approval_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and Scaler saved!")


import streamlit as st
import joblib
import numpy as np

model = joblib.load("loan_approval_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Loan Approval Prediction")

# User Inputs
Gender = st.selectbox("Gender", [0, 1])  # Encoded
Married = st.selectbox("Married", [0, 1])
Dependents = st.selectbox("Dependents", [0, 1, 2, 3])
Education = st.selectbox("Education", [0, 1])
Self_Employed = st.selectbox("Self Employed", [0, 1])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
Credit_History = st.selectbox("Credit History", [0, 1])
Property_Area = st.selectbox("Property Area", [0, 1, 2])

if st.button("Predict"):
    TotalIncome = ApplicantIncome + CoapplicantIncome
    features = np.array([[Gender, Married, Dependents, Education, Self_Employed,
                          ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
                          Credit_History, Property_Area, TotalIncome]])
    features = scaler.transform(features)
    prediction = model.predict(features)
    st.success("Loan Approved" if prediction[0] == 1 else "Loan Rejected")
