# Employee Salary Prediction Using Machine Learning

A Machine Learning web application that predicts whether an employee falls
into the **High** or **Low** salary category, based on age, education level,
experience, income, and outstanding loan amount. Built with a Random Forest
Classifier and served through a Flask web app.

---

## Project Overview

Employee Salary Prediction is a Machine Learning project that classifies
employees into salary categories using a trained Random Forest model. The
goal of the project is to automate salary band estimation and assist
organizations in making data-driven compensation decisions.

The project uses **Python**, **scikit-learn**, and **Flask** to analyze
employee data, train a classification model, and serve real-time predictions
through a web interface.

## Problem Statement

Manually classifying employees into salary brackets can be time-consuming
and inconsistent across an organization. This project builds a machine
learning model that predicts an employee's salary category from their
attributes — age, education, experience, income, and loan amount — and
exposes that model through a simple web app.

## Objectives

- Generate and analyze employee data.
- Train a Random Forest classifier for salary category prediction.
- Evaluate the model with standard classification metrics.
- Serve predictions through a Flask web application.
- Demonstrate an end-to-end ML deployment workflow.

## Technologies Used

- Python
- Pandas / NumPy
- Scikit-learn
- Joblib
- Flask
- HTML / CSS (Jinja2 templates)

## Project Structure

```
employee_salary_pred/
├── app.py                  # Flask web application
├── generate_dataset.py     # Generates the synthetic employee dataset
├── train_model.py          # Trains, evaluates, and saves the model
├── requirements.txt
├── data/
│   └── employee_data.csv   # Generated dataset (1,000 records)
├── model/
│   ├── trained_model.joblib    # Trained RandomForestClassifier
│   └── label_encoder.joblib    # Encodes "High"/"Low" labels
├── templates/
│   └── index.html          # Web form + result UI
└── static/
    └── style.css
```

## Project Workflow

### 1. Data Collection

`generate_dataset.py` builds a synthetic but realistic dataset of 1,000
employee records with the following features:

- `age`
- `education_level` (0 = High School, 1 = Bachelors, 2 = Masters, 3 = PhD)
- `experience` (years)
- `income` (annual, INR)
- `loan_amount` (outstanding, INR)
- `salary_category` (target: **High** / **Low**)

Income is generated as a function of age, education, and experience (plus
noise), and `salary_category` is derived from a median income split, with
5% label noise added to avoid a trivially separable dataset.

### 2. Data Preprocessing

- Numeric features are used directly (no missing values, since the data is
  generated programmatically).
- The target column `salary_category` is label-encoded ("High"/"Low" → 0/1)
  using `LabelEncoder`.

### 3. Feature Selection

```
age
education_level
experience
income
loan_amount
```

### 4. Model Training

The dataset is split into:

- Training set (80%)
- Testing set (20%, stratified)

A `RandomForestClassifier` (`n_estimators=200`, `max_depth=8`) is trained on
the training set.

### 5. Model Evaluation

The model is evaluated on the held-out test set using accuracy, a full
classification report (precision/recall/F1), and a confusion matrix. On the
generated dataset this achieves **~95% accuracy**.

### 6. Model Saving

```python
joblib.dump(model, "model/trained_model.joblib")
joblib.dump(label_encoder, "model/label_encoder.joblib")
```

### 7. Deployment

`app.py` loads the saved model and label encoder, and serves:

- `/` — an HTML form to enter employee details
- `/predict` — handles form submission and renders the predicted category
- `/api/predict` — a JSON API endpoint for programmatic predictions

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset
python generate_dataset.py

# 3. Train the model
python train_model.py

# 4. Run the web app
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

### Using the JSON API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 32, "education_level": 1, "experience": 8, "income": 64000, "loan_amount": 15000}'
```

Response:

```json
{"salary_category": "High", "confidence": 91.5}
```

## Machine Learning Logic

**Input** — employee attributes, e.g.:

```
age = 32
education_level = 1   (Bachelors)
experience = 8
income = 64000
loan_amount = 15000
```

**Processing** — the Random Forest algorithm:

1. Builds multiple decision trees on bootstrapped samples of the training data.
2. Each tree independently predicts a salary category.
3. The final prediction is determined by majority vote across all trees.

**Output**

```
salary_category = "High"   or   "Low"
```

along with a confidence score (the model's predicted probability for the
winning class).

## Algorithm Used: Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines many decision
trees to improve prediction accuracy and reduce overfitting compared to a
single tree.

**Advantages:**

- High accuracy on structured/tabular data
- Handles non-linear relationships well
- Reduces overfitting via averaging across trees
- Provides interpretable feature importances

On this dataset, `income` is by far the strongest predictor (as expected,
since the label is income-derived), followed by `experience` and `age`.

## Future Enhancements

- Replace the synthetic dataset with a real-world employee dataset.
- Add more granular salary bands instead of a binary High/Low split.
- Compare Random Forest against other algorithms (Logistic Regression,
  Gradient Boosting, XGBoost).
- Add input validation and authentication for production deployment.
- Add a visualization dashboard (feature importance, prediction history).
- Containerize with Docker and deploy to a cloud platform.

## Conclusion

This project demonstrates an end-to-end machine learning workflow: synthetic
data generation, preprocessing, model training and evaluation, model
persistence with Joblib, and deployment behind a Flask web interface. The
Random Forest model learns patterns relating employee attributes to salary
category and serves predictions in real time through both a web form and a
JSON API.

## Short Project Logic (for Viva / Interview)

We generated a structured employee dataset using Pandas and NumPy, with
features (age, education level, experience, income, loan amount) and a
target salary category derived from income. The dataset was split into
training and testing sets (80/20, stratified). A Random Forest Classifier
was trained on the training data to learn the relationship between employee
attributes and salary category, achieving ~95% test accuracy. The trained
model and its label encoder were saved using Joblib, then loaded inside a
Flask application that exposes both an HTML form and a JSON API for making
predictions on new employee records in real time.

---

**Author:** Pooja S — [github.com/pooja-s6](https://github.com/pooja-s6)
