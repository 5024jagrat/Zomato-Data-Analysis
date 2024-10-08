Python_Project: Zomato Analysis Project
Overview
This project involves the analysis of Zomato restaurant data using Streamlit and Power BI. The provided Python code generates an interactive web application to visualize and explore the dataset. Leveraging the Streamlit, pandas, and Plotly libraries, the app offers an engaging user interface for exploring relationships between different variables. The application provides various visualizations such as bar charts, 3D scatter plots, clustered bar charts, heatmaps, line charts, scatter plots, and a 3D plot using Plotly.

Dependencies
Ensure that you have the following Python packages installed before running the application:

streamlit
pandas
plotly
scikit-learn
matplotlib
You can install these packages using the following command:

bash
Copy code
pip install streamlit pandas plotly scikit-learn matplotlib
Data Source
The project uses Zomato restaurant data, which is loaded from a CSV file. Make sure the CSV file (data.csv) is available at the specified location in the code. The data is cleaned and processed for visualization.

python
Copy code
# Load Zomato data
import pandas as pd
zomato_data = pd.read_csv("E:/Python/data.csv")
Data Cleaning
(Include the data cleaning steps here as per your code)

User Interface (UI)
The Streamlit web application has a user-friendly interface with the following features:

Dropdowns for selecting X-Axis, Y-Axis, and Z-Axis data attributes.
Numeric input fields for specifying city_id, average_cost_for_two, price_range, aggregate_rating, votes, and photo_count.
A "Predict" button to perform predictions using a pre-trained random forest model.
Different tabs for various visualizations, such as bar charts, 3D scatter plots, clustered bar charts, heatmaps, line charts, scatter plots, and 3D visualizations using Plotly.
Server (Backend)
The server-side code contains functions to render various types of plots based on user inputs. Additionally, it includes an event handler for the "Predict" button, which utilizes a pre-trained random forest model to make predictions based on user input.

How to Run
To run the Zomato Analysis Python application, execute the following command in the terminal:

bash
Copy code
streamlit run path_to_directory_containing_code/app.py
Replace "path_to_directory_containing_code" with the actual path to the directory where your code is saved.

Predictions
The application allows users to input values for city_id, average_cost_for_two, price_range, aggregate_rating, votes, and photo_count and click the "Predict" button to obtain predictions for aggregate_rating. Ensure that the pre-trained random forest model file is loaded before running the application.

Here’s how to load the model in Python:

python
Copy code
import joblib

# Load the pre-trained model
model = joblib.load("E:/Python/ml_model.pkl")
Make sure that the ml_model.pkl file contains the pre-trained random forest model.


## Screen Shots
![Screenshot 2023-12-09 110415](https://github.com/Mehul1611/R_Project/assets/111687116/42ec1f63-2000-4d81-af3d-c3e77ca08433)
![Screenshot 2023-12-09 110426](https://github.com/Mehul1611/R_Project/assets/111687116/a4d2c567-4462-4707-9e4a-9375cc84dbb1)
![Screenshot 2023-12-09 110437](https://github.com/Mehul1611/R_Project/assets/111687116/686c6711-eba5-49c6-9c2e-ee1330930d64)
![Screenshot 2023-12-09 110507](https://github.com/Mehul1611/R_Project/assets/111687116/0500fc3e-5e4c-4bf2-91b9-7f44e58187cd)
![Screenshot 2023-12-09 110521](https://github.com/Mehul1611/R_Project/assets/111687116/cd909390-8cd5-47b8-a488-4c071e6e751b)
![Screenshot 2023-12-09 110444](https://github.com/Mehul1611/R_Project/assets/111687116/cc78f2be-8d7a-4dfe-b4c2-02292364789e)
![Screenshot 2023-12-09 110451](https://github.com/Mehul1611/R_Project/assets/111687116/a00c9e7f-14a6-4880-90a8-7618ddd2504a)
![Screenshot 2023-12-09 111946](https://github.com/Mehul1611/R_Project/assets/111687116/b9aaa7a9-778f-4ce4-910a-02130775cd43)

## Disclaimer

This project is provided as a demonstration and may require customization based on your specific needs. The accuracy of predictions depends on the quality and representativeness of the training data used to create the random forest model.


## Contributors
-[Mehul Sharma](https://github.com/Mehul1611)
-[Madhav Somani](https://github.com/Somanimadhav)

