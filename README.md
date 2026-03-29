# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import required libraries and load the dataset.
2. Preprocess the data and split into training and testing sets.
3. Apply standardization and create pipelines with polynomial features.
4. Train Ridge, Lasso, and ElasticNet models using training data.
5. Predict results and evaluate performance using MSE and R² score.
## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: POOJA U
RegisterNumber: 25011745/212225230209

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("encoded_car_data (1).csv")

# Clean column names (important fix)
data.columns = data.columns.str.strip()

print(data.head())

# Drop unnecessary columns if present
data = data.drop(['CarName', 'car_ID'], axis=1, errors='ignore')

# Split features and target
X = data.drop('price', axis=1)
y = data['price']

# Train-test split (before scaling)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Scale target properly
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# Models
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}

results = {}

# Training loop
for name, model in models.items():

    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results[name] = {'MSE': mse, 'R2_Score': r2}

# Print output
print("\nName: POOJA U")
print("Reg No: 25011745/212225230209")

for model_name, metrics in results.items():
    print(f"\n{model_name} Model")
    print(f"MSE: {metrics['MSE']:.2f}")
    print(f"R2 Score: {metrics['R2_Score']:.2f}")

# Convert to DataFrame
results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)

# Plot graphs
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.barplot(x='Model', y='MSE', data=results_df)
plt.title("Mean Squared Error Comparison")
plt.xticks(rotation=45)

plt.subplot(1,2,2)
sns.barplot(x='Model', y='R2_Score', data=results_df)
plt.title("R2 Score Comparison")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

*/
```

## Output:

<img width="1225" height="721" alt="Screenshot 2026-03-29 222950" src="https://github.com/user-attachments/assets/6fd88e93-cd2c-4daa-b349-20a2505c38b7" />

<img width="1325" height="610" alt="Screenshot 2026-03-29 223003" src="https://github.com/user-attachments/assets/3f242c2f-81d7-43e5-b6dc-6c24fea5f40b" />

## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
