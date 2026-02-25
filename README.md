# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize weights, bias, learning rate, and epochs.
2. Update weights using SGD for each training sample.
3. Predict outputs using the trained model.
4. Compute error metrics to evaluate performance.

## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: POOJA U
RegisterNumber: 212225230209
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso,ElasticNet
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv("encoded_car_data (1).csv")
data.head()
df = pd.get_dummies(data, drop_first=True)

X = data.drop('price',axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)


scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)

models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5)
}

results ={}

for name,model in models.items():
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('regressor', model)
    ])

pipeline.fit(X_train,y_train)
pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, pred)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test,pred)

results[name] = {'MSE' : mse, 'MAE' : mae, 'Rscore': r2}

print("Name: POOJA U")
print("Reg. No: 212225230209")
for model_name, metrics in results.items():
    print(f"{model_name} - \nMean Squared Error: {metrics['MSE']:.2f}, \nMean Absolute Error: {metrics['MAE']:.2f}, \nR Squared Score: {metrics['Rscore']:.2f}")

results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'},inplace=True)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.barplot(x='Model',y='MSE',data=results_df, palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)

plt.subplot(1,2,2)
sns.barplot(x='Model',y='Rscore', data=results_df,palette='viridis')
plt.title('R Squared Score')
plt.ylabel('R Squared Score')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
*/
```

## Output:

<img width="1313" height="731" alt="Screenshot 2026-02-25 090248" src="https://github.com/user-attachments/assets/2215b9c2-b0a6-4dce-b77b-c940b116f1e0" />
<img width="1320" height="762" alt="Screenshot 2026-02-25 090303" src="https://github.com/user-attachments/assets/1318f951-a03e-4837-b7ad-5993dfecef01" />


## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
