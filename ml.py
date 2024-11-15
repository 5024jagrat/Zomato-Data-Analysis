# Load required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Data Preprocessing and Cleaning
df = pd.read_csv("data.csv")
df = df.drop(columns=['url', 'city', 'res_id', 'delivery', 'takeaway', 'establishment', 'latitude', 'longitude', 
                      'country_id', 'cuisines', 'rating_text', 'name', 'address', 'locality', 'zipcode', 'locality_verbose', 
                      'timings', 'currency', 'highlights', 'opentable_support'])
df = df.dropna()

# Define X and Y for model
X = df.drop(columns=['aggregate_rating'])
Y = df['aggregate_rating']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

# Fit a random forest model
model = RandomForestRegressor(random_state=123)
model.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = model.predict(X_test)

# Calculate the accuracy for the model
accuracy = 1 - (mean_squared_error(Y_test, Y_pred) / ((Y_test - Y_test.mean())**2).sum())
print(f"Accuracy: {accuracy}")

# Prediction example
example_predictions = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
print(example_predictions.head())

# Print the data types of each column
column_data_types = df.dtypes
print(column_data_types)
