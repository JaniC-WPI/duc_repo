import numpy as np

# Example array, replace this with your actual data
data =  [19, 47, -79,  7, 89, -13, 71, -59, -23, 61, 41, -67,  -3,  2, -11, 29, -43, -31, 73, 17, -5, 83, 37, 53]


# Reshape the data into a matrix, here I'm assuming it's a 4x6 matrix
matrix = np.array(data).reshape(8, 3)
print(matrix)

# Calculate the rank using SVD
u, s, vh = np.linalg.svd(matrix)
rank = np.sum(s > 1e-4)

print(f"Rank of the matrix: {rank}")

import numpy as np
from sympy import primerange

# Generate a list of prime numbers between 1 to 100
prime_numbers = list(primerange(1, 91))

# Randomly select 24 primes from the list without replacement
selected_primes = np.random.choice(prime_numbers, size=24, replace=False)

# Ensure the numbers are spread out well and not repeating in the same row or column
# This is achieved by reshaping the selected primes into an 8x3 matrix
jacobian_matrix = selected_primes

print(jacobian_matrix)