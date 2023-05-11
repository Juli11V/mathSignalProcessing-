import numpy as np
import time
import matplotlib.pyplot as plt

def fft(x):
    """
    Функція для обчислення ШПФ для вхідного сигналу x за допомогою рекурсивного алгоритму.
    """
    n = len(x)
    if n == 1:
        return x
    else:
        # Рекурсивно обчислюємо ШПФ для парних та непарних елементів вхідного сигналу
        m = n // 2
        even = fft(x[::2])
        odd = fft(x[1::2])
        # Обчислюємо W-матрицю, яка складається з експонент з множником -2πj/N, де N - довжина вхідного сигналу
        W = np.exp(-2j * np.pi * np.arange(m) / n)
        # Об'єднуємо результати обчислень для парних та непарних елементів
        return np.concatenate([even + W * odd, even - W * odd])

def fft_iter(x):
    """
    Функція для обчислення ШПФ для вхідного сигналу x за допомогою ітеративного алгоритму.
    """
    n = len(x)
    levels = int(np.log2(n))
    # Завдання початкових значень
    X = x.astype(complex)
    for level in range(levels):
        # Розбиття вектора на дві частини
        step = 2 ** level
        for k in range(step):
            # Обчислення W-матриці
            W = np.exp(-2j * np.pi * k / (2 * step))
            # Обчислення ШПФ для кожного з піввекторів та їх об'єднання
            for i in range(k, n, 2 * step):
                t = W * X[i + step]
                X[i + step] = X[i] - t
                X[i] += t
    return X

# Генеруємо випадковий сигнал довжиною 16
N = 16
x = np.random.rand(N)
# Доповнюємо вхідний сигнал нулями до степеня 2
M = 2**int(np.ceil(np.log2(N)))
x = np.concatenate([x, np.zeros(M-N)])

# Обчислюємо ШПФ за допомогою рекурсивного алгоритму
t1 = time.time()
X = fft(x)
t2 = time.time()

# Виводимо результати
for i, val in enumerate(X):
    print(f"C_{i}: {val}")
# вивід часу обчислення
print(f"\nЧас виконання: {t2 - t1:.6f} секунд")

# обрахунок кількості операцій
num_plus = N
num_mult = 4 * N
num_operations = num_plus + num_mult
print(f"\nКількість операцій множення та додавання: {num_operations}")

# обчислення спектру амплітуд і фаз для вхідного вектору
amp = np.abs(X)
phase = np.angle(X)

# побудова графіку спектру амплітуд і фаз
plt.figure()
plt.stem(amp)
plt.title("Амплітудний спектр")
plt.xlabel("Частота")
plt.ylabel("Амплітуда")

plt.figure()
plt.stem(phase)
plt.title("Фазовий спектр")
plt.xlabel("Частота")
plt.ylabel("Фаза")

plt.show()
