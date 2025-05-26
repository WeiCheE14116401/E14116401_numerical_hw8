import numpy as np

# 數據
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

# 矩陣 A
A = np.vstack([np.ones_like(x), x, x**2]).T

# 係數
coeffs = np.linalg.lstsq(A, y, rcond=None)[0]

# 擬合值
y_fit = A @ coeffs

# 誤差
error = np.sum((y - y_fit)**2)

print(f"係數: {coeffs}")
print(f"誤差: {error}")
