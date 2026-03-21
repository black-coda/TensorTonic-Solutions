from typing import List

def median(values: List[int]):
    data = sorted(values)
    n = len(data)
    mid = n // 2

    if n % 2 == 0:
        return (data[mid - 1] + data[mid]) / 2
    else:
        return data[mid]


def calculate_iqr(values):
    data = sorted(values)
    n = len(data)
    mid = n // 2

    # Split the data into two halves
    if n % 2 == 0:
        lower_half = data[:mid]
        upper_half = data[mid:]
    else:
        # Exclude the exact middle element for the halves
        lower_half = data[:mid]
        upper_half = data[mid+1:]

    q1 = median(lower_half)
    q3 = median(upper_half)

    return q3 - q1

def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    if len(values) == 1:
        return [0.0]
    n = len(values)
    q2 = median(values)
    iqr = calculate_iqr(values)

    

    for i in range(len(values)):
        
        try:
            x_scaled = (values[i] - q2)/iqr
        except ZeroDivisionError:
            x_scaled = (values[i] - q2)  
        values[i] = x_scaled
    return values


