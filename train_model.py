"""
train_model.py
---------------
Trains a Random Forest Classifier to predict employee salary category
(High / Low) based on age, education level, experience, income, and
loan amount.

Run this script after generate_dataset.py:
    python generate_dataset.py
    python train_model.py

Produces:
    model/trained_model.joblib   - the trained classifier
    model/label_encoder.joblib   - encodes "High"/"Low" <-> 1/0
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ---------------------------------------------------------
# Step 1: Load Dataset
# ---------------------------------------------------------
df = pd.read_csv("data/employee_data.csv")
print("Dataset shape:", df.shape)
print(df.head())

# ---------------------------------------------------------
# Step 2: Prepare Features and Target
# ---------------------------------------------------------
FEATURE_COLUMNS = ["age", "education_level", "experience", "income", "loan_amount"]

X = df[FEATURE_COLUMNS]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["salary_category"])  # High=0/Low=1 or vice versa

print("Label classes:", list(label_encoder.classes_))

# ---------------------------------------------------------
# Step 3: Train-Test Split (80/20)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# Step 4: Train Random Forest Classifier
# ---------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# Step 5: Evaluate Model
# ---------------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance (useful for README / viva explanation)
importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
print("\nFeature Importances:")
print(importance_df.to_string(index=False))

# ---------------------------------------------------------
# Step 6: Save Trained Model and Label Encoder
# ---------------------------------------------------------
joblib.dump(model, "model/trained_model.joblib")
joblib.dump(label_encoder, "model/label_encoder.joblib")
print("\nModel saved to model/trained_model.joblib")
print("Label encoder saved to model/label_encoder.joblib")
