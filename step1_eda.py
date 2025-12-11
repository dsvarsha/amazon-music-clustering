import pandas as pd

# Load the dataset
df = pd.read_csv("single_genre_artists.csv")

# 1. Show first 5 rows
print("\n--- HEAD (first 5 rows) ---")
print(df.head())

# 2. Shape of the dataset
print("\n--- SHAPE (rows, columns) ---")
print(df.shape)

# 3. Column names
print("\n--- COLUMNS ---")
print(df.columns)

# 4. Data types of each column
print("\n--- DATA TYPES ---")
print(df.dtypes)

# 5. Missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 6. Duplicate rows
print("\n--- DUPLICATES ---")
print(df.duplicated().sum())
