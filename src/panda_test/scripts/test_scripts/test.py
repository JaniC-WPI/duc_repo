import numpy as np

# Example array, replace this with your actual data
data =  [31, 13, 53, 47, 79, 13, 23, 73,  7, 83, 41, 79, 61, 73, 59, 61, 71, 17,  5, 11, 31, 11, 31, 31,
 71, 67, 53, 71, 11, 67]


# Reshape the data into a matrix, here I'm assuming it's a 4x6 matrix
matrix = np.array(data).reshape(10, 3)
print(matrix)

# Calculate the rank using SVD
u, s, vh = np.linalg.svd(matrix)
rank = np.sum(s > 1e-4)

print(f"Rank of the matrix: {rank}")

import numpy as np
from sympy import primerange

# Generate a list of prime numbers between 1 to 114
prime_numbers = list(primerange(1, 175))

# Randomly select 24 primes from the list without replacement
selected_primes = np.random.choice(prime_numbers, size=36, replace=False)

# Ensure the numbers are spread out well and not repeating in the same row or column
# This is achieved by reshaping the selected primes into an 8x3 matrix
jacobian_matrix = selected_primes

print(jacobian_matrix)