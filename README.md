Titanic Survival Prediction

This project predicts the survival of passengers on the Titanic using machine learning models. It is a part of learning data preprocessing, feature engineering, and model building in Python.

🧰 Tools & Libraries

Python 3.x

Pandas

NumPy

Scikit-learn

Matplotlib / Seaborn (optional for visualization)

📂 Dataset

The dataset contains information about Titanic passengers, including:

PassengerId: Unique ID

Pclass: Ticket class

Name, Sex, Age

SibSp: Number of siblings/spouses aboard

Parch: Number of parents/children aboard

Ticket, Fare, Cabin, Embarked

The dataset is publicly available on Kaggle Titanic Competition
.

⚙️ Features

Some important features engineered for this model:

Title: Extracted from passenger names

FamilySize: Combination of SibSp + Parch

IsAlone: Whether passenger was traveling alone

Age imputation and binning

Fare binning

Sex and Embarked converted to numeric

💡 Model

We used Random Forest Classifier for predictions. The model was trained and validated on a train-test split of the dataset. Hyperparameters like n_estimators and max_depth were tuned for better accuracy.

📈 Accuracy

The current model achieves around [insert your accuracy]% on the validation set.

⚠️ Note: The model is still being improved by adding more feature engineering and hyperparameter tuning.

📦 How to Run

Clone this repository:

git clone https://github.com/yourusername/titanic-survival-prediction.git


Install required libraries:

pip install -r requirements.txt


Run the Jupyter Notebook or Python script:

jupyter notebook Titanic_Survival.ipynb

🔮 Future Improvements

Try other models like XGBoost, Gradient Boosting

Feature scaling and additional feature engineering

Cross-validation for more robust evaluation

📌 References

Kaggle Titanic Competition

Scikit-learn Documentation

