# Step 1: Understanding Machine Learning
# Machine learning (ML) is about teaching computers to learn patterns from data and make predictions. There are three types of ML:
# 1. Supervised Learning - The model learns from labeled data (e.g. prediction house prices based on size, locations, etc).
# 2. Unsupervised Learning - The model finds patterns in unlabeled data(e.g., customer segmentation, market basket analysis, etc). customer segementation is grouping customers based on their purchase history.
# 3. Reinforcement Learning - The model learns by trial and error (e.g., training a self-driving car to drive on the road). The model learns from the consequences of its actions.

# for now, we will focus on supervised learning as it is the most commonly used ML type.

# Step 2: Install Scikit-learn
# Scikit-learn is a popular ML library in Python. It provides a wide range of tools for building ML models. You can install scikit-learn using pip:
# pip install scikit-learn

# Step 3: Build a Simple Machine Learning Model

# let's create a simple model to predict house prices based on square footage.

# Dataset (House Prices)
# Suppose we have a dataset of house prices and their square footage. The dataset looks like this:
# | Square Footage | Price |
# |----------------|-------|
# | 1000           | 300000|
# | 1500           | 450000|    
# | 2000           | 600000|
# | 2500           | 750000|
# | 3000           | 900000|

# Step 4: Write Python Code

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Prepare the dataset
square_feet = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)  # features (x) 
# reshape(-1, 1) is used to convert 1D array to 2D array and -1 is used to infer the number of rows based on the number of columns. 
# In this case, we have 1 column, so the number of rows is inferred automatically. converting 1D array to 2D array is required by scikit-learn and it means that we have 5 samples and 1 feature.

prices = np.array([300000, 450000, 600000, 750000, 900000])  # target (y)

# 2. Train the model
model = LinearRegression()  # create a linear regression model
model.fit(square_feet, prices)  # train the model on the dataset

# 3. Make predictions
new_size = np.array([2200]).reshape(-1, 1)  # predict the price of a house with 2200 square feet
predicted_price = model.predict(new_size)
print(f"The predicted price of a house with 2200 square feet is ${predicted_price[0]:.2f}")

# 4. Visualize the results
plt.scatter(square_feet, prices, color='blue', label="Actual Data")  # plot the actual data
plt.plot(square_feet, model.predict(square_feet), color='red', label="Regression Line")  # plot the regression line
plt.scatter(new_size, predicted_price, color='green', marker='x', s=100, label="Predicted Price")  # plot the predicted price
plt.xlabel("Square Footage")
plt.ylabel("Price ($1000s)")
plt.title("House Price Prediction")
plt.legend()
plt.show()

# run the code and you should see the following output: 
# The predicted price of a house with 2200 square feet is $550000.00

# Step 5: Interpret the Results
# The model predicts the price of a house with 2200 square feet to be $550,000. 
# The red line is the regression line that best fits the data points. 
# The green cross represents the predicted price of the house. 
# The model has learned the relationship between square footage and price and can make accurate predictions based on new data.
