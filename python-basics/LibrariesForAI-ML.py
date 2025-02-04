# Description: This file contains the libraries that are essential for AI and ML projects.

# 1. NumPy
# NumPy is essential for working with arrays and matrices. It provides support for large, multi-dimensional arrays and provides many mathematical functions for manipulating these arrays.

# to install numpy, run the following command:
# pip install numpy

# basic NumPy operations

import numpy as np

# create an array
arr = np.array([1, 2, 3, 4, 5])

# Array operations
print(arr + 2) # add 2 to each element in the array
print(arr * 2) # multiply each element by 2
print(arr - 2) # subtract 2 from each element
print(arr / 2) # divide each element by 2

# 2. Pandas
# Pandas is used for data manipulation and analysis. It provides two main classes: Series (1-dimensional) and DataFrame (2-dimensional). It is particularly useful when dealing with data in tabular
# form, such as CSV files.

# to install pandas, run the following command:
# pip install pandas

# Example usage of Pandas

import pandas as pd

# create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']}

df = pd.DataFrame(data)  # create a DataFrame from a dictionary
# DataFrames are used to store data in tabular form, tabular data is data that is stored in rows and columns, like a spreadsheet. Used cases include reading data from CSV files, SQL tables, or other data sources.

# Viewing the DataFrame
print(df)

# Accessing columns
print(df['Name'])  # access the 'Name' column

# Accessing multiple columns
print(df[['Name', 'City']])  # access the 'Name' and 'City' columns 

# Accessing rows and columns
print(df.loc[0, 'Name'])  # access the 'Name' column of the first row  loc means location of the row and column
# the output will be: Alice

# Accessing rows
print(df.iloc[0])  # access the first row  iloc means index location of the row  and here we are not specifying the column so it will return all the columns of the first row 
# the output will be: Name Alice Age 25 City New York

# Adding a new column
df("job") = ['Engineer', 'Doctor', 'Teacher', 'Lawyer']

# Filtering data
print(df[df['Age'] > 30])  # filter rows where Age is greater than 30
# the usage of df, df is the dataframe and df['Age'] > 30 is the condition that we are checking for and the output will be the rows where the age is greater than 30  
# the output will be: Name Age City job 2 Charlie 35 Chicago Teacher 3 David 40 Houston Lawyer


# 3. Matplotlib and Seaborn
# These are two important libraries for data visualization. Matplotlib is a plotting library that provides a MATLAB-like interface for creating plots and charts. 
# Seaborn is built on top of Matplotlib and provides a higher-level interface for creating attractive and informative statistical graphics.
# They allow you to plot graphs and charts, which are essential for understanding data in AI and ML projects.

# to isntall matplotlib, run the following command:
# pip install matplotlib

# Basic usage of Matplotlib

import matplotlib.pyplot as plt

# creating a simple plot
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

plt.plot(x, y)  # plot the data
plt.xlabel('x-axis')  # set the label for the x-axis
plt.ylabel('y-axis')  # set the label for the y-axis
plt.title('Simple Plot')  # set the title of the plot
plt.show()  # display the plot 
# this will display a simple line plot with x-axis values 1, 2, 3, 4, 5 and y-axis values 2, 3, 5, 7, 11

# to install seaborn, run the following command:
# pip install seaborn

# Basic usage of Seaborn

import seaborn as sns

# load a sample dataset
tips = sns.load_dataset('tips') # load the 'tips' dataset from the seaborn library

# display the first few rows of the dataset
print(tips.head())

# create a scatter plot
sns.scatterplot(x='total_bill', y='tip', data=tips)  # create a scatter plot of 'total_bill' vs 'tip'
plt.show()  # display the plot 
# this will display a scatter plot of 'total_bill' vs 'tip' using the 'tips' dataset