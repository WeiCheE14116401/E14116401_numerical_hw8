import math
import numpy as np
PI = np.pi

def f(x):
    return x**2 * math.sin(x)

def S4(x):
    a0 = 0.459205
    a = [-0.146756, 0.054608, -0.038929, 0.033542]
    b = [0.232287, -0.124941, 0.082932, 0]
    
    total = a0
    for k in range(1, 5):  # k=1~4
        angle = k * PI * (2*x - 1)
        total += a[k-1] * math.cos(angle)
        total += b[k-1] * math.sin(angle)
    return total

def calculate_error():
    m = 32
    error = 0.0
    
    for j in range(m):
        xj = j / m
        diff = f(xj) - S4(xj)
        error += diff**2
    
    return error

if __name__ == "__main__":
    error = calculate_error()
    print(f"E(S4) = {error}")
