# Load required libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Data Preprocessing and Cleaning
df = pd.read_csv("E://R//data.csv")
# Drop unnecessary columns
df = df.drop(columns=['url', 'city', 'res_id', 'delivery', 'takeaway', 'establishment', 
                      'latitude', 'longitude', 'country_id', 'cuisines', 'rating_text', 
                      'name', 'address', 'locality', 'zipcode', 'locality_verbose', 
                      'timings', 'currency', 'highlights', 'opentable_support'])

# Drop missing values
df = df.dropna()

# Define X and Y for the model
X = df.drop(columns=["aggregate_rating"])
Y = df["aggregate_rating"]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

# Fit a Random Forest model
model = RandomForestRegressor()
model.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = model.predict(X_test)

# Calculate the accuracy for the model (R^2 score)
accuracy = r2_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy}")

# Prediction example
example_predictions = pd.DataFrame({"Actual": Y_test, "Predicted": Y_pred})
print(example_predictions.head())

# Find the data type of each column
column_data_types = df.dtypes
print(column_data_types)
