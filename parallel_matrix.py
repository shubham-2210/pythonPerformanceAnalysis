import numpy as np
import time
from multiprocessing import Process, cpu_count
from threading import Thread

def matrix_multiply(size=500):
    """Performs matrix multiplication on two random matrices of given size."""
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    return np.dot(A, B)

def parallel_multiply_threading(size=500, num_threads=4):
    """Parallelizes matrix multiplication using threading."""
    threads = []
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    result = np.zeros((size, size))

    def worker(start, end):
        result[start:end] = np.dot(A[start:end], B)

    chunk_size = size // num_threads
    for i in range(num_threads):
        start = i * chunk_size
        end = size if i == num_threads - 1 else (i + 1) * chunk_size
        thread = Thread(target=worker, args=(start, end))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return result

def parallel_multiply_multiprocessing(size=500, num_processes=4):
    """Parallelizes matrix multiplication using multiprocessing."""
    processes = []
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    result = np.zeros((size, size))

    def worker(start, end, result_slice):
        result_slice[:] = np.dot(A[start:end], B)

    chunk_size = size // num_processes
    result_slices = np.zeros((num_processes, chunk_size, size))
    
    for i in range(num_processes):
        start = i * chunk_size
        end = size if i == num_processes - 1 else (i + 1) * chunk_size
        process = Process(target=worker, args=(start, end, result_slices[i]))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    result = np.vstack(result_slices)
    return result

if __name__ == "__main__":
    size = 500

    print("Baseline (CPython) Execution:")
    start_time = time.time()
    matrix_multiply(size)
    print(f"Execution time: {time.time() - start_time:.3f} seconds")

    print("\nThreading Execution:")
    start_time = time.time()
    parallel_multiply_threading(size)
    print(f"Execution time: {time.time() - start_time:.3f} seconds")

    print("\nMultiprocessing Execution:")
    start_time = time.time()
    parallel_multiply_multiprocessing(size)
    print(f"Execution time: {time.time() - start_time:.3f} seconds")
