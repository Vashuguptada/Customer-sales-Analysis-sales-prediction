# Customer-sales-Analysis-sales-prediction
Customer sales Analysis &amp; sales prediction


## README

### Project Overview

This project focuses on predicting sales based on monthly and yearly features using various machine learning models. The script involves data preparation, model evaluation, sales prediction, and results exportation.

### Prerequisites

Ensure you have Python 3.7+ and install the required libraries using pip:

```sh
pip install pandas numpy scikit-learn xgboost
```

### Data Preparation

1. **Load the Data**:
   Load the dataset from a CSV file named `fake_data.csv` with columns:
   - Order id, Email, Sales, Date, Product quantity, Product name, Product sku, Customer Name, Customer City, Customer Zip, Customer Phone.

2. **Preprocess the Data**:
   ```python
   import pandas as pd
   df = pd.read_csv('fake_data.csv')
   df['Date'] = pd.to_datetime(df['Date'])
   df.set_index('Date', inplace=True)
   df['Month'] = df.index.month
   df['Year'] = df.index.year
   ```

3. **Define Features and Target**:
   ```python
   X = df[['Month', 'Year']]
   y = df['Sales']
   ```

4. **Split the Data**:
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   ```

### Model Evaluation

Define a function to evaluate models:

```python
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred
```

Initialize and evaluate models:

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=0),
    'Random Forest': RandomForestRegressor(random_state=0),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=0)
}

results = {}
for name, model in models.items():
    mse, r2, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {'MSE': mse, 'R-squared': r2, 'Predictions': y_pred}
```

### Best Model Selection

Select the best model based on the R-squared metric:

```python
best_model_name = max(results, key=lambda k: results[k]['R-squared'])
best_model = models[best_model_name]
```

### Sales Prediction

Predict next month's sales for each customer:

```python
import numpy as np

last_billing_dates = df.reset_index().groupby(['Email', 'Customer Name'])['Date'].max()
predictions = []

for (email, customer_name), last_date in last_billing_dates.items():
    last_sales = df.loc[df.index == last_date, 'Sales'].values[0]
    next_month = (last_date.month % 12) + 1
    next_year = last_date.year if next_month > last_date.month else last_date.year + 1
    next_month_features = np.array([[next_month, next_year]])
    predicted_sales = best_model.predict(next_month_features)
    predictions.append((email, customer_name, last_date, last_sales, next_month, next_year, predicted_sales[0]))

predictions_df = pd.DataFrame(predictions, columns=['Email', 'Customer Name', 'Last_Billing_Date', 'Last_Sales', 'Next_Month', 'Next_Year', 'Predicted_Sales'])
predictions_df.to_csv('customer_next_month_predictions.csv', index=False)
```

### Results

Print results and export predictions:

```python
print(f"Best model: {best_model_name}")
print(predictions_df.head())
```

### Conclusion

This project uses multiple regression models to predict sales based on date features. The script selects the best-performing model and predicts future sales for each customer, saving the results to a CSV file.
