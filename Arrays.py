# Import necessary libraries
import numpy as np                      # NumPy for array and matrix operations
import matplotlib.pyplot as plt         # Matplotlib (not used here, but imported for plotting if needed)

print("Working with Arrays")
print("="*70)                           # Print a line of 70 '=' characters for formatting

# -------------------------------
# Creating Arrays
# -------------------------------
arr1 = np.array([[1, 2], [3, 4]])       # Create first 2x2 array
arr2 = np.array([[5, 6], [7, 8]])       # Create second 2x2 array

print(f"\nArray 1 (2x2):\n{arr1}")       # Display arr1
print(f"Array 2 (2x2):\n{arr2}")        # Display arr2

# -------------------------------
# Array Reshape
# -------------------------------
arr3 = np.arange(12).reshape(3, 4)      # Create array with 0–11 and reshape to 3x4 matrix
print(f"Array 3 (3x4):\n{arr3}")

# -------------------------------
# Arithmetic Operations
# -------------------------------
print("\nArithmetic Operations")
print("-"*50)

print(f"Addition:\n{arr1 + arr2}")      # Element-wise addition
print(f"Subtract: \n{arr1 - arr2}")     # Element-wise subtraction
print(f"Multiply: \n{arr1 * arr2}")     # Element-wise multiplication
print(f"Divide: \n{arr1 / arr2}")       # Element-wise division
print(f"Power: \n{arr1 ** 2}")          # Square every element of arr1

# -------------------------------
# Array Properties
# -------------------------------
print("\nArray Properties")
print("-"*50)

print(f"Shape arr1: {arr1.shape}")      # Shape → (rows, columns)
print(f"Shape arr3: {arr3.shape}")      
print(f"Dtype arr1: {arr1.dtype}")      # Data type
print(f"Size arr1: {arr1.size}")        # Total number of elements
print(f"ndim arr1: {arr1.ndim}")        # Number of dimensions

# -------------------------------
# Aggregate Functions
# -------------------------------
print("\nAggregrate Functions")
print("-"*50)

print(f"Sum: {np.sum(arr3)}")           # Sum of all elements in arr3
print(f"Mean: {np.mean(arr3):.2f}")     # Mean value
print(f"Min/Max: {np.min(arr3)}, {np.max(arr3)}")  # Min and Max value
print(f"Std: {np.std(arr3):.2f}")       # Standard deviation
print(f"Row sums: {np.sum(arr3, axis=1)}")   # Sum across each row
print(f"Col sums: {np.sum(arr3, axis=0)}")   # Sum across each column

# -------------------------------
# Array Manipulation
# -------------------------------
print("\nArray Manipulation")
print("-"*50)

print(f"Reshape: {np.arange(12).reshape(2, 6)}")    # Reshape into 2x6 array
print(f"Flatten: {arr3.ravel()}")                   # Convert to 1D array
print(f"Transpose:{arr3.T.shape}")                  # Transpose (flip rows and columns)

# -------------------------------
# Universal Functions (ufuncs)
# -------------------------------
print("\nUniversal Functions")
print("-"*50)

print(f"Sqrt: {np.sqrt(arr1)}")                     # Square root
print(f"Exp: {np.exp(arr1[:1])}")                   # Exponential function
print(f"Sin: {np.sin(arr3[:3])}")                   # Sine function for each element

# -------------------------------
# Matrix Operations
# -------------------------------
print("\nMATRIX Operations")
print("-"*50)

print(f"Dot product:\n{np.dot(arr1, arr2)}")        # Dot product
print(f"Matrix mult:\n{np.matmul(arr1, arr2)}")     # Matrix multiplication

# -------------------------------
# Statistical Operations
# -------------------------------
print("\nStatisticl Operations")
print("-"*50)

print(f"Median: {np.median(arr3):.2f}")             # Median value
print(f"Percentile 25: {np.percentile(arr3, 25):.2f}")  # 25th percentile

print("="*70)
