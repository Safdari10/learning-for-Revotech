# Data Structures in Python
# below are the data structures that frequently use when processing data in Python

# 1. List 
# A list is an ordered collection of items. It can contain items of different data types and you can accessor modify these items using an indeces(index).

# List of numbers
numbers = [1, 2, 3, 4, 5]

# Accessing elements in a list
print(numbers[0]) # 1

# Modify elements in a list
numbers[1] = 200 # [1, 200, 3, 4, 5] 

# Adding elementsto a list
numbers.append(300) # [1, 200, 3, 4, 5, 300]

# Slicing a list
print(numbers[1:3]) # [200, 3] because it starts at index 1 and stops at index 3


# 2. Dictionaries
# A dictionary is a collection of key-value pairs. You can store data by associating a key with a value, making it easier to access. 

# Dictionary example
person  = {"name": "John", "age": 30, "city": "New York"}

# Accessing elements in a dictionary
print(person["name"]) # John

# Modifying elements in a dictionary
person["age"] = 40 # {"name": "John", "age": 40, "city": "New York"}
person["country"] = "USA" # {"name": "John", "age": 40, "city": "New York", "country": "USA"}

# Removing elements from a dictionary
del person["city"] # {"name": "John", "age": 40, "country": "USA"}

# looping through a dictionary
for key, value in person.items():
    print(f"key: value") # key: name, key: age, key: country
    #f-string is used to format the output of the print statement

# 3. Tuples
# A tuple is similar to a list but it is immutable. You can't modify the elements in a tuple once it is created.

# Tuple example
coordinates = (10, 20)

# Accessing elements in a tuple
print(coordinates[0]) # 10

# 4. Sets
# A set is a collection of unique elements. You can perform set operations like union, intersection, difference, and symmetric difference.

# Set example
unique_numbers = {1, 2, 3, 4, 5}

# Adding elements to a set
unique_numbers.add(6) # {1, 2, 3, 4, 5, 6}

# Removing elements from a set
unique_numbers.remove(3) # {1, 2, 4, 5, 6}

# Set operations
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}
set3 = set1.union(set2) # {1, 2, 3, 4, 5, 6, 7, 8}
set4 = set1.intersection(set2) # {4, 5}
set5 = set1.difference(set2) # {1, 2, 3} why because it returns the elements that are in set1 but not in set2
set6 = set1.symmetric_difference(set2) # {1, 2, 3, 6, 7, 8} it returns the elements that are in set1 or set2 but not in both

# 5. Stacks
# A stack is a data structure that follows the Last In First Out (LIFO) principle. You can add elements to the top of the stack and remove elements from the top of the stack.

# Stack example
stack = []
stack.append(1) # [1]
stack.append(2) # [1, 2]
stack.append(3) # [1, 2, 3]
stack.pop() # [1, 2]


# 6. Queues
# A queue is a data structure that follows the First In First Out (FIFO) principle. You can add elements to the back of the queue and remove elements from the front of the queue.

# Queue example
queue = []
queue.append(1) # [1]
queue.append(2) # [1, 2]
queue.append(3) # [1, 2, 3]
queue.pop(0) # [2, 3]



