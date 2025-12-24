import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load the dataset
# Make sure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in your folder
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Data Cleaning
# Convert TotalCharges to numeric (handle errors by turning them into NaN, then filling with 0)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# 3. Encoding Categorical Variables
# Machine learning models only understand numbers
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object' and column != 'customerID':
        df[column] = le.fit_transform(df[column])

# 4. Define Features (X) and Target (y)
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# 5. Split data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Build and Train the Model (Random Forest is great for this)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 9. Visualization: Feature Importance
# This shows what causes customers to leave (Contract, Tenure, etc.)
plt.figure(figsize=(10,6))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Factors Influencing Customer Churn')
plt.show()