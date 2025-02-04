import pandas as pd
import matplotlib.pyplot as plt
import os

# step 1: Read the CSV file
df = pd.read_csv('people.csv')

# step 2: Display the first few rows of the dataframe
print("Dataset preview:\n", df.head())

# step 3: Calculate the average age
average_age = df['age'].mean()
print(f"Average age: {average_age:.2f}") # Display the average age with 2 decimal places 

# sort the dataframe by age in descending order
df = df.sort_values(by='age', ascending=False)

# step 4: Plot a bar chart of ages
plt.bar(df['name'], df['age'], color='skyblue')
plt.xlabel('Name')
plt.ylabel('Age')
plt.title('Age of Individuals')
plt.xticks(rotation=45) # Rotate the x-axis labels for better readability
plt.show() 


# step 5: Save the sorted dataframe to a new CSV file
if 'sorted_people.csv' in os.listdir():
    os.remove('sorted_people.csv') # Remove the file if it already exists
df.to_csv('sorted_people.csv', index=False) 

# step 6: Plot a pie chart of the number of individuals in each city
city_count = df['city'].value_counts()
city_count.plot.pie(autopct='%1.1f%%', startangle=90) # Display percentage values with 1 decimal place
plt.title('Distribution of Individuals by city')
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()