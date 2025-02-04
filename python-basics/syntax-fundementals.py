# 1. Variables and Data Types

# integer
age = 25

# float
height = 5.9

# string
name = "John"

# boolean
is_student = True

# we can perform basic operations on these types.

#Arithmetic with numbers
sum_result = 10 + 5 # 15
product_result = 10 * 5 # 50
difference_result = 10 - 5 # 5
quotient_result = 10 / 5 # 2
remainder_result = 10 % 5 # 0
exponential_result = 10 ** 5 # 100000

#Concatenation with strings
greeting = "Hello, " + name + "!" # Hello, John!

# boolean operations
is_adult = age > 18 # True


#2. Control Flow Statements (if, elif, else)

# if statement
if age >= 18:
    print("You are an adult.")
else:
    print("You are a minor.")
    
# Elif for multiple conditions
if height > 6:
    print("You are tall.")
elif height > 5:
    print("You are average height.")
else:
    print("You are short.")

#3. Loops (for, while)

# For loop - iterate over a list
colors = ["red", "green", "blue"]
for color in colors: 
    print(color)
    
# While loop - repeat until a condition is met/false
counter = 0

while counter < 5:
    print(counter)
    counter += 1 # increment the counter by 1
    

#4. Functions

# Defining a function
def greet(name):
    return "Hello, " + name + "!"

# Calling a function
message = greet("Alice")
print(message) # Hello, Alice!

# Functions can have default arguments
def greet(name, message="Hello"):
    return message + ", " + name + "!"

message = greet("Alice", "Hi")

#5. Lists, Dictonaries, Tuples, and Sets

# List: Ordered collection of items, can be modified, and can contain duplicates (mutable)
fruits = ["apple", "banana", "cherry"]
fruits.append("date") # add an item to the end of the list
print(fruits[0]) # apple

# Dictionary: Unordered collection of key-value pairs (mutable)
person = {"name": "Alice", "age": 25}
print(person["name"]) # Alice

# Tuple: Ordered collection of items, cannot be modified (immutable)
coordinates = (10, 20)
print(coordinates[0]) # 10

# Set: Unordered collection of unique items (mutable)
unique_numbers = {1, 2, 3, 3, 3}
print(unique_numbers) # {1, 2, 3} 
# Sets are useful for finding unique items in a list


# Practice
# 1. Write a function that takes a list of numbers as input and returns the sum of all even numbers in the list.
numbers = [1 , 2, 3, 4, 5, 6, 7, 8, 9, 10]

def sum_even(numbers):
    sum = 0
    for number in numbers:
        if number % 2 == 0:
            sum += number
    return sum