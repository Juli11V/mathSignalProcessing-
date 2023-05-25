import numpy as np
import matplotlib.pyplot as plt

def calculate_arithmetic_mean(values):
    return np.mean(values)

def calculate_harmonic_mean(values):
    return len(values) / np.sum(1 / values)

def calculate_geometric_mean(values):
    values = np.where(values <= 0, np.nan, values)
    return np.nanprod(values) ** (1 / np.sum(~np.isnan(values)))

def generate_test_sequence(N, A, n, phi, max_error_ratio):
    x = np.linspace(0, 1, N)
    y_exact = A * np.sin(n * x + phi)
    max_error = max_error_ratio * A
    errors = np.random.uniform(-max_error, max_error, N)
    y_distorted = y_exact + errors
    return x, y_exact, y_distorted

def calculate_exact_value(x, A, n, phi):
    return A * np.sin(n * x + phi)

def compare_values(approximate, exact):
    absolute_error = np.abs(approximate - exact)
    relative_errors = absolute_error / np.abs(exact)
    return absolute_error, relative_errors

def plot_result(x, y, y_exact):
    plt.plot(x, y, label='Спотворене значення')
    plt.plot(x, y_exact, label='Точне значення')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

N = 6000
A = 1
n = 6
phi = 0
max_error_ratio = 0.05

x, y_exact, y_distorted = generate_test_sequence(N, A, n, phi, max_error_ratio)
arithmetic = calculate_arithmetic_mean(y_distorted)
harmonic = calculate_harmonic_mean(y_distorted)
geometric = calculate_geometric_mean(y_distorted)

print("\nСереднє арифметичне:", arithmetic)
print("Середнє гармонічне:", harmonic)
print("Середнє геометричне:", geometric)

y_exact = calculate_exact_value(x, A, n, phi)
y_exact[np.abs(y_exact) < np.finfo(float).eps] = np.finfo(float).eps
absolute_errors, relative_errors = compare_values(y_distorted, y_exact)
max_absolute_error = np.max(absolute_errors)
min_absolute_error = np.min(absolute_errors)
max_relative_error = np.max(relative_errors)
min_relative_error = np.min(relative_errors)

print("\nМаксимум абсолютної похибки:", max_absolute_error)
print("Мінімум абсолютної похибки:", min_absolute_error)
print("Мінімум відносної похибки:", min_relative_error)

plot_result(x, y_distorted, y_exact)
