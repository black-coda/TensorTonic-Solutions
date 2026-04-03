from typing import Optional

def max_n_dim(arr: np.ndarray, axis: Optional[int] = None):
    row, col = arr.shape
    
    if axis is None:
        return np.max(arr)
    
    if axis == 0:  # For columns (operate along rows)
        print("Max for each column:")
        for j in range(col):
            col_max = np.max(arr[:, j])
            print(f"  Column {j}: {col_max}")
        return np.max(arr, axis=0)
    
    elif axis == 1:  # For rows (operate along columns)
        print("Max for each row:")
        for i in range(row):
            row_max = np.max(arr[i, :])
            print(f"  Row {i}: {row_max}")
        return np.max(arr, axis=1).reshape(-1,1)


def l1(matrix: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    L1 distance between two matrices.
    """
    if axis == 1:
        ax_sum = np.sum(np.abs(matrix), axis=axis).reshape(-1,1)
        print(f"Shape: {ax_sum.shape} \n {ax_sum}")
    else:
        ax_sum = np.sum(np.abs(matrix), axis=axis)
    return np.divide(matrix, ax_sum,where=(ax_sum != 0), out=np.zeros_like(matrix, dtype=float))

def l2(matrix: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    L2 distance between two matrices.
    """
    if axis == 1:
        ax_sum = np.sqrt(np.sum(np.square(matrix), axis=axis)).reshape(-1,1)
        print(f"Shape: {ax_sum.shape} \n {ax_sum}")
    else:
        ax_sum = np.sqrt(np.sum(np.square(matrix), axis=axis))
    return np.divide(matrix, ax_sum,where=(ax_sum != 0), out=np.zeros_like(matrix, dtype=float))

def max_norm(matrix: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    L2 distance between two matrices.
    """
    ax_sum = max_n_dim(matrix,axis=axis)
    print(f'max_norm: {ax_sum.shape}')
    return np.divide(matrix, ax_sum,where=(ax_sum != 0), out=np.zeros_like(matrix, dtype=float))



def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    np.where(condition, value_if_true, value_if_false)
    """
    matrix = np.asarray(matrix)
    matrix = np.where(np.isnan(matrix),0,matrix)
    if matrix.ndim == 1 or matrix.ndim > 2:
        return None
    if axis not in [None, 1,0]:
        return None
        
    print(matrix.ndim)
    # Write code here
    if norm_type == 'l1':
        return l1(matrix, axis=axis)
    elif norm_type == "l2":
        return l2(matrix, axis=axis)
    elif norm_type == "max":
        return max_norm(matrix, axis=axis)
    else:
        return None
