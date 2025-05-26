import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# 原始函數
def f(x):
    return 0.5 * np.cos(x) + 0.25 * np.sin(2*x)

# Legendre多項式
def legendre_p0(x):
    return np.ones_like(x) #1

def legendre_p1(x):
    return x

def legendre_p2(x):
    return 0.5 * (3 * x**2 - 1) #(3x^2-1)/2

# 計算係數
c0 = 0.5 * integrate.quad(lambda x: f(x) * legendre_p0(x), -1, 1)[0]
c1 = 1.5 * integrate.quad(lambda x: f(x) * legendre_p1(x), -1, 1)[0]
c2 = 2.5 * integrate.quad(lambda x: f(x) * legendre_p2(x), -1, 1)[0]

# 產生近似多項式
def p_legendre(x):
    return c0 * legendre_p0(x) + c1 * legendre_p1(x) + c2 * legendre_p2(x)

# 轉換為標準多項式形式
standard_form_coeffs = [c2 * 1.5, c1, c0 - 0.5 * c2]

print(f"Legendre係數: c0 = {c0:.8f}, c1 = {c1:.8f}, c2 = {c2:.8f}")
print(f"標準多項式形式: {standard_form_coeffs[0]:.8f}x^2 + {standard_form_coeffs[1]:.8f}x + {standard_form_coeffs[2]:.8f}")

# 計算近似誤差
error = integrate.quad(lambda x: (f(x) - p_legendre(x))**2, -1, 1)[0]
print(f"近似誤差: {error:.8f}")
