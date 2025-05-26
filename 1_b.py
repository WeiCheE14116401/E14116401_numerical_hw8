import numpy as np
# 數據
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

# 對數轉換
Y = np.log(y)

# 線性設計矩陣
A_exp = np.vstack([np.ones_like(x), x]).T

# 解係數
coeffs_exp = np.linalg.lstsq(A_exp, Y, rcond=None)[0]

# 計算擬合值並轉回原尺度
Y_fit = A_exp @ coeffs_exp
y_fit_exp = np.exp(Y_fit)

# 計算誤差
error_exp = np.sum((y - y_fit_exp)**2)

# 將係數轉換為原始參數
b = np.exp(coeffs_exp[0])
a = coeffs_exp[1]

print(f"b = {b}, a = {a}")
print(f"誤差: {error_exp}")
