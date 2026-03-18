import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    B = np.zeros((len(A[0]), len(A)), dtype=int)
    rows = len(A)
    cols = len(A[0])
    for i in range(rows):

        for j in range(cols):
            print(f"i is {i} and j is {j}")
            # print(f"A[{i}][{j}] is {A[i][j]} and A[{j}][{i}] is {A[j][i]}")
            B[j][i] = A[i][j]

            # print(f'Swapping A[{i}][{j}] and A[{j}][{i}]')
    return B