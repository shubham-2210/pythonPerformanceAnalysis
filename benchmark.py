import time
import numpy as np

def matrix_multiply(size=500):
    """Performs matrix multiplication on two random matrices of given size."""
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    return np.dot(A, B)

if __name__ == "__main__":
    start_time = time.time()
    matrix_multiply()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.3f} seconds")
