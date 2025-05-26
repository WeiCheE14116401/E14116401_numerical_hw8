import numpy as np

# 數據
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

# 取對數轉換
X_log = np.log(x)  #ln(x)
Y_log = np.log(y)  #ln(y)

# 線性矩陣
A_pow = np.vstack([np.ones_like(X_log), X_log]).T

# 求係數
coeffs_pow = np.linalg.lstsq(A_pow, Y_log, rcond=None)[0]

# 計算擬合值並轉回原尺度(exp)
Y_fit_pow = A_pow @ coeffs_pow
y_fit_pow = np.exp(Y_fit_pow)

# 計算誤差
error_pow = np.sum((y - y_fit_pow)**2)

# 將係數轉換為原始參數
b_pow = np.exp(coeffs_pow[0])
n_pow = coeffs_pow[1]

print(f"b = {b_pow}, n = {n_pow}")
print(f"誤差: {error_pow}")
