
# THIS FILE GENERATES RANDOM DATA ABOUT THE AVAILABLE CAPACITY IN DIFFERENT SERVICES

import random
import csv

column_numbers = [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51]
real_capacity = [5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50]

# Read data from the CSV file
with open('MY_CSV.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

# Extract data from the specified columns corresponding to real_capacity
extracted_data = []
for row in data:
    row_data = [int(row[index]) for index in real_capacity]
    extracted_data.append(row_data)

# Update values in the specified columns with random numbers centered around the extracted data
for row in data:
    for col_num, capacity in zip(column_numbers, extracted_data):
        # Generate a random number within a range centered around the capacity
        haha = capacity / 4  # Access individual capacity value from the list
        range_min = max(0, haha)  # Lower bound of the range
        range_max = min(capacity + 5, capacity * 2)  # Upper bound of the range
        new_value = random.randint(range_min, range_max)
        row[col_num] = new_value

# Print the updated data
for row in data:
    print(row)

# Write the updated data back to the CSV file
with open('MY_CSV.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)
