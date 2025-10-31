# 🌾 Crop Recommendation using Random Forest (Modified Version)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import joblib

# ===============================================
# 1️⃣ Load the dataset
# ===============================================
# Replace the path below with the location of your own Crop_recommendation.csv file
dataset_path = r"C:\Users\Sahana B M\Downloads\CropRecommendationProject\Crop_recommendation.csv"
crop_data = pd.read_csv(dataset_path)

print("✅ Dataset successfully loaded!")
print(f"Total Records: {crop_data.shape[0]}, Total Features: {crop_data.shape[1]}")
print("\n🔹 First 5 rows of data:")
print(crop_data.head())

# ===============================================
# 2️⃣ Check for missing values
# ===============================================
print("\nChecking for missing values...")
missing = crop_data.isnull().sum()
print(missing if missing.sum() > 0 else "No missing values found!")

# ===============================================
# 3️⃣ Feature selection
# ===============================================
# Independent Variables
inputs = crop_data.iloc[:, :-1]   # all columns except the last one
# Dependent Variable
output = crop_data.iloc[:, -1]    # last column (label)

# ===============================================
# 4️⃣ Split the dataset into Train and Test
# ===============================================
X_train, X_test, y_train, y_test = train_test_split(
    inputs, output, test_size=0.2, random_state=42
)

print(f"\nData Split Completed:")
print(f"Training Samples: {X_train.shape[0]}, Testing Samples: {X_test.shape[0]}")

# ===============================================
# 5️⃣ Model Training
# ===============================================
rf_model = RandomForestClassifier(
    n_estimators=120, random_state=42, criterion='gini'
)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, "trained_crop_model.pkl")
print("\n💾 Model has been saved as 'trained_crop_model.pkl'")

# ===============================================
# 6️⃣ Model Evaluation
# ===============================================
y_predicted = rf_model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_predicted)
print(f"\n📊 Model Accuracy: {acc*100:.2f}%")

# Classification Report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_predicted))

# Confusion Matrix
cmatrix = confusion_matrix(y_test, y_predicted)
print("\nConfusion Matrix:\n", cmatrix)

# ===============================================
# 7️⃣ Confusion Matrix Visualization
# ===============================================
plt.figure(figsize=(9, 6))
sns.heatmap(
    cmatrix, annot=True, fmt='d', cmap='YlGnBu',
    xticklabels=rf_model.classes_, yticklabels=rf_model.classes_
)
plt.title("🌾 Crop Prediction Confusion Matrix")
plt.xlabel("Predicted Crop")
plt.ylabel("Actual Crop")
plt.tight_layout()
plt.show()
