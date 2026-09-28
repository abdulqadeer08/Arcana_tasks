# Python and Data Foundations

## Python data types
Python has several built-in data types. Integers (int) store whole numbers, floats store decimal numbers, strings (str) store text, and booleans (bool) store True or False. Lists are ordered, mutable collections written with square brackets. Tuples are ordered but immutable. Dictionaries store key-value pairs and give fast lookup by key. Sets store unique, unordered items and are useful for removing duplicates and for membership tests.

## Functions and scope
A function is defined with the def keyword and groups reusable logic. Variables created inside a function are local to that function. Variables defined at the top level of a module are global. Python resolves names using the LEGB rule: Local, Enclosing, Global, Built-in.

## NumPy and Pandas
NumPy provides the ndarray, a fast fixed-type array that supports vectorised operations and broadcasting, which avoids slow Python loops. Pandas builds on NumPy and provides the DataFrame, a table with labelled rows and columns. Common Pandas operations include read_csv to load data, head to preview rows, groupby to aggregate, merge to join tables, and fillna or dropna to handle missing values.

## Descriptive statistics
The mean is the average value, the median is the middle value after sorting, and the mode is the most frequent value. The median is robust to outliers while the mean is not. Variance measures how spread out values are around the mean, and the standard deviation is the square root of the variance. Correlation measures the strength of a linear relationship between two variables and ranges from -1 to 1. Correlation does not imply causation.
