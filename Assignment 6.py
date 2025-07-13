# 1. Import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# 2. Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 3. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Models to evaluate
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

# 6. Evaluate base models
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print("📊 Base Model Performance:")
print(results_df)

# 7. Hyperparameter tuning

# GridSearchCV for Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring='f1')
grid_rf.fit(X_train, y_train)

# RandomizedSearchCV for SVM
from scipy.stats import uniform
svm_params = {
    'C': uniform(0.1, 10),
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}
rand_svm = RandomizedSearchCV(SVC(), svm_params, n_iter=10, cv=5, scoring='f1', random_state=42)
rand_svm.fit(X_train, y_train)

# Logistic Regression tuning (optional)
log_params = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
grid_log = GridSearchCV(LogisticRegression(max_iter=1000), log_params, cv=5, scoring='f1')
grid_log.fit(X_train, y_train)

# 8. Evaluate tuned models
tuned_models = {
    'Tuned Logistic Regression': grid_log.best_estimator_,
    'Tuned Random Forest': grid_rf.best_estimator_,
    'Tuned SVM': rand_svm.best_estimator_
}

tuned_results = []

for name, model in tuned_models.items():
    y_pred = model.predict(X_test)
    tuned_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred)
    })

tuned_df = pd.DataFrame(tuned_results)
print("\n🚀 Tuned Model Performance:")
print(tuned_df)

# 9. Select best model based on F1-score
best_model = tuned_df.sort_values(by='F1-score', ascending=False).iloc[0]
print(f"\n🏆 Best Model: {best_model['Model']} with F1-score: {best_model['F1-score']:.4f}")
