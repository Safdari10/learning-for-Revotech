import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('dataset.csv')

# step 1: Prepare the dataset
X = data.iloc[:, :-1].values  # get all the columns except the last one
y = data.iloc[:, -1].values  # get the last column as the target column

# step 2: Train the model
model = LinearRegression()
model.fit(X, y)

# step 3: Make predictions
new_house = np.array([[895, 3, 3]]) # 895 sqft, 3 bedrooms, 3 years old
predicted_price = model.predict(new_house)
print(f"The predicted price for 895 sqft, 3 bedrooms, 3 years old house is ${predicted_price[0]:.2f}")

# step 4: Visualize the predictions (for 2D, we pick one variable to compare)
plt.scatter(X[:, 0], y, color='blue', label="Actual Prices") # sqft vs price
plt.scatter(new_house[0, 0], predicted_price, color='red', marker='x', s=100, label="Predicted Price") # sqft vs predicted price
plt.xlabel("Square Feet")
plt.ylabel("Price (in $1000s)")
plt.title("House Price Prediction with Multiple Features")
plt.legend()  # show the legend: legend is the label of the plot elements and in this case it is "Actual Prices" and "Predicted Price"
plt.show() # show the plot

# step 5: Evaluate the model
r_squared = model.score(X, y) # R-squared value is a measure of how well the model fits the data
print(f"R-squared value: {r_squared:.2f}")  # closer to 1 is better

