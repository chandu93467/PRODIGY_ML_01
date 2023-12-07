# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Sample dataset
data = {
    'SquareFootage': [1500, 2000, 2500, 3000, 1800],
    'Bedrooms': [3, 4, 3, 4, 3],
    'Bathrooms': [2, 2.5, 3, 2, 2],
    'Price': [200000, 250000, 300000, 350000, 220000]
}

df = pd.DataFrame(data)

# Split the data into features (X) and target variable (y)
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Ridge Regression model
alpha = 1.0  # You can adjust the regularization strength (alpha) based on your data
ridge_model = Ridge(alpha=alpha)

# Train the model
ridge_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = ridge_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Print the coefficients and intercept
print('Coefficients:', ridge_model.coef_)
print('Intercept:', ridge_model.intercept_)
