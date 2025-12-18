# Titanic Survival Prediction 🛳️
Predicting passenger survival on the Titanic using machine learning and feature engineering techniques.


# Project Overview

This project aims to predict whether a passenger survived the Titanic disaster using machine learning. The project demonstrates:

Data cleaning and preprocessing

Feature engineering

Model training and evaluation

# Dataset
Kaggle Titanic Dataset

The dataset includes passenger information from the Titanic, such as:

PassengerId, Pclass, Name, Sex, Age

SibSp (siblings/spouses aboard), Parch (parents/children aboard)

Ticket, Fare, Cabin, Embarked

Source: Kaggle Titanic Dataset

Features

Some key engineered features used for modeling:

Title – extracted from passenger names

FamilySize – SibSp + Parch

IsAlone – passenger traveling alone or not

Age – imputed and binned

Fare – binned

Sex and Embarked – converted to numeric values

# Model

We used a Random Forest Classifier for predictions.
Hyperparameters such as n_estimators and max_depth were tuned for better accuracy.

Accuracy on validation set: 77.99%

⚠️ Note: Model is still being improved with additional feature engineering and tuning.

# Installation

Clone the repository:

git clone https://github.com/Kr1v/titanic-survival-prediction.git
cd titanic-survival-prediction


Install required libraries:

pip install -r requirements.txt

Usage

Run the Jupyter Notebook or Python script:

jupyter notebook Titanic.ipynb

# ⚡ Tech Stack

Python | Pandas | NumPy | Scikit-learn 

# Future Improvements

Experiment with other models: XGBoost, Gradient Boosting

Additional feature engineering

# References

Kaggle Titanic Competition

Scikit-learn Documentation

