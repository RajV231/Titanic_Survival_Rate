# Titanic Survival Prediction Project - Final Enhanced Version

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Provided Titanic dataset
data = pd.read_csv('Titanic-Dataset.csv')

# Basic EDA (Exploratory Data Analysis)
print("Basic Info:\n", data.info())
print("\nMissing Values:\n", data.isnull().sum())
print("\nDataset Sample:\n", data.head())

# Visualizations
sns.countplot(x='Survived', data=data)
plt.title("Survival Count")
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title("Survival by Passenger Class")
plt.show()

sns.countplot(x='Sex', hue='Survived', data=data)
plt.title("Survival by Gender")
plt.show()

# ========================
# Data Cleaning & Feature Engineering
# ========================

data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# New Feature: Title extracted from Name (unique!)
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')

# Map titles to numbers
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
data['Title'] = data['Title'].map(title_mapping)
data['Title'].fillna(0, inplace=True)

# Drop columns not useful for prediction
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# New Feature: Family Size
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# Encode Categorical Variables
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
data['Embarked'] = le.fit_transform(data['Embarked'])

# Feature Set and Target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Random Forest with GridSearchCV for better tuning
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
}
rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid=params, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

# Predictions
y_pred = grid.predict(X_test)

# Evaluation
print("Best Parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance Plot
feature_imp = pd.Series(grid.best_estimator_.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.title('Feature Importance')
plt.show()
