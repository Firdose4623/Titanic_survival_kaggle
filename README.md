# 🚢 Titanic Survival Prediction - Machine Learning Project

Predicting survival on the Titanic using real data. This classic binary classification problem was solved using end-to-end ML techniques, including EDA, preprocessing, feature engineering, model building, hyperparameter tuning, and evaluation.

---

## 📁 Dataset

- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)
- File used:
  - `train.csv`

---

## 📌 Problem Statement

The goal is to build a machine learning model that can predict whether a passenger survived or not, based on features such as age, gender, class, family relationships, and more.

---

## 🔍 Exploratory Data Analysis (EDA)

- Identified missing values and feature distributions
- Visualized survival patterns based on:
  - Gender
  - Passenger class
  - Embarkation point
  - Family size
  - Fare amount
  - Age

---

## 🛠️ Data Preprocessing & Feature Engineering

- Imputed missing values (`Age`, `Fare`, `Embarked`)
- Created new feature: `FamilySize = SibSp + Parch + 1`
- One-hot encoded categorical variables (`Sex`, `Embarked`, `Pclass`)
- Scaled continuous variables
- Dropped irrelevant columns (`Name`, `Ticket`, `Cabin`, `PassengerId`)

---

## 🤖 Models Trained

| Model                   | Status          |
|-------------------------|------------------|
| Logistic Regression     | ✅ Baseline model |
| Random Forest Classifier| ✅ Best performer |
| XGBoost                 | ✅ Compared       |

---

## 🧪 Hyperparameter Tuning

Performed using **GridSearchCV** on Random Forest:

```python
param_grid = {
  'n_estimators': [100, 200],
  'max_depth': [None, 5, 10],
  'min_samples_split': [2, 5],
  'min_samples_leaf': [1, 2]
}
```

- 5-fold cross-validation
- Evaluated using F1-score
- Helped improve generalization

---

## 📈 Evaluation Metrics (Best Model: Random Forest)

- **Accuracy**: `80%`
- **Confusion Matrix**:
  ```
  [[98 12]
   [23 46]]
  ```

- **Precision**:
  - Class 0 (Did not survive): `0.81`
  - Class 1 (Survived): `0.79`

- **Recall**:
  - Class 0: `0.89`
  - Class 1: `0.67`

- **F1-Score**:
  - Class 0: `0.85`
  - Class 1: `0.72`

- **Macro Avg F1-Score**: `0.79`
- **Weighted Avg F1-Score**: `0.80`

➡️ The model performs better at identifying passengers who did not survive (Class 0), but recall for survivors (Class 1) could be improved with more balanced data or feature engineering.

---

## ✅ Key Learnings

- Real-world EDA and handling missing data
- Feature engineering based on domain understanding
- Tried and tuned multiple machine learning models
- Used evaluation metrics for informed model selection

---

## 🚀 Project Status

✔️ Completed core pipeline  
📊 Tuned and evaluated models  
🧠 Ready for GitHub & LinkedIn  
⏳ Test set & Kaggle submission planned for later

---

## 📬 Connect with Me

Feel free to reach out or follow:

- 💼 [LinkedIn](https://www.linkedin.com/in/firdose-anjum-ml)
- 🧠 Portfolio (coming soon)

---

## 🧠 Tools & Technologies

- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

> If you like this project, ⭐ star it and follow me for more ML content!
