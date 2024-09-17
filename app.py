from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Initialize Flask App
app = Flask(__name__)

# Load Zomato data
zomato_data = pd.read_csv("E:/R/data.csv")
zomato_data = zomato_data.drop(columns=['res_id', 'url', 'address', 'locality', 'zipcode', 'locality_verbose',
                                        'cuisines', 'timings', 'currency', 'highlights', 'opentable_support'])

zomato_data.columns = ["Name", "Establ", "City", "City_id", "Lat", "Long", "C_ID", "Cost", "Price", 
                       "Agg", "Rating", "Votes", "Count", "Delivery", "Takeaway"]

# Train a random forest model
X = zomato_data[['City_id', 'Cost', 'Price', 'Votes', 'Count']]
Y = zomato_data['Rating']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, Y_train)

# Flask route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Flask route to predict using the model
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    city_id = float(data['city_id'])
    cost = float(data['average_cost_for_two'])
    price_range = float(data['price_range'])
    rating = float(data['aggregate_rating'])
    votes = int(data['votes'])
    photo_count = int(data['photo_count'])

    # Prepare data for prediction
    input_data = np.array([[city_id, cost, price_range, votes, photo_count]])
    prediction = model.predict(input_data)[0]
    
    return jsonify({'prediction': round(prediction, 2)})

# Flask route to create charts using Plotly
@app.route('/plot', methods=['POST'])
def plot():
    data = request.json
    x_axis = data['x_axis']
    y_axis = data['y_axis']

    if x_axis != "None" and y_axis != "None":
        fig = px.bar(zomato_data, x=x_axis, y=y_axis, title=f'Bar Chart: {y_axis} by {x_axis}')
        return jsonify({'plot': fig.to_html(full_html=False)})

    return jsonify({'plot': ''})

# Main function to run the app
if __name__ == "__main__":
    app.run(debug=True)
