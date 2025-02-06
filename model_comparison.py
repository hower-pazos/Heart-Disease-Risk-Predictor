import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from joblib import dump
from sklearn.tree import DecisionTreeClassifier

# Load data
data = pd.read_csv('heart.csv')

# Handle duplicates and missing data
data = data.drop_duplicates()
X = data.drop(columns=['target'])  
y = data['target'] 

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Feature selection
selector = SelectKBest(score_func=f_classif, k='all')
X_selected = selector.fit_transform(X_imputed, y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models and Hyperparameter Tuning
models = {
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
}

param_grids = {
    'SVM': {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100], 'gamma': ['auto', 'scale']},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20], 'min_samples_split': [2, 5]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
    'Naive Bayes': {'var_smoothing': [1e-9, 1e-8, 1e-7]},
    'Decision Tree': {'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10]},
}

# Initialize dictionary to store metrics
metrics = {}

# Training and Tuning Each Model
for model_name, model in models.items():
    print(f"\nTraining and tuning {model_name}...")
    
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best {model_name} parameters: {grid_search.best_params_}")
    
    # Evaluating model on Train and Test data
    y_train_pred = grid_search.predict(X_train_scaled)
    y_test_pred = grid_search.predict(X_test_scaled)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"{model_name} - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Storing metrics in dictionary
    metrics[model_name] = {
        "Precision": classification_report(y_test, y_test_pred, output_dict=True)['1']['precision'] * 100,
        "Recall": classification_report(y_test, y_test_pred, output_dict=True)['1']['recall'] * 100,
        "F1-Score": classification_report(y_test, y_test_pred, output_dict=True)['1']['f1-score'] * 100,
        "Accuracy": test_accuracy * 100
    }

# Stacking Classifier
print("\nTraining Stacking Classifier...")
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10)),
    ('svc', SVC(kernel='linear', C=10))
]
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=AdaBoostClassifier())
stacking_model.fit(X_train_scaled, y_train)

y_train_pred = stacking_model.predict(X_train_scaled)
y_test_pred = stacking_model.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Stacking Classifier - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print("Classification Report:")
print(classification_report(y_test, y_test_pred))

# Storing metrics for Stacking Classifier
metrics["Stacking Classifier"] = {
    "Precision": classification_report(y_test, y_test_pred, output_dict=True)['1']['precision'] * 100,
    "Recall": classification_report(y_test, y_test_pred, output_dict=True)['1']['recall'] * 100,
    "F1-Score": classification_report(y_test, y_test_pred, output_dict=True)['1']['f1-score'] * 100,
    "Accuracy": test_accuracy * 100
}

# Display the metrics
for model_name, model_metrics in metrics.items():
    print(f"{model_name}: {model_metrics}")

# Best model (Naive Bayes) chosen based on test accuracy
best_model = GaussianNB()
best_model.fit(X_train_scaled, y_train)

# Save the best model to a file
dump(best_model, 'best_model.pkl')

