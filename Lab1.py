import numpy as np
import matplotlib.pyplot as plt
import math
import os
import scipy.integrate as integrate

# 1 Завдання

# Межі
li = -3*np.pi
lf = 3*np.pi
# Порядок ряду Фур'є
n = 30

# Створюємо масив x значень від -π до π з кроком 0.01
x = np.arange(li, lf, 0.1)

# Обчислюємо значення функції f(x) на масиві x
y = x**6 * np.exp(-x**2/6)

# Побудова графіку функції
plt.plot(x, y)

# Встановлюємо межі графіку
plt.xlim([li, lf])
plt.ylim([-10, 70])

# Додаємо підписи для осей та заголовок графіку
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Графік функції f(x) = x^6 · exp(-x^2/6) на інтервалі [-3π, 3π]')

# Відображаємо графік
plt.show()


# Завдання 2

# Визначення функції

def fourier(li, lf, x, n):
    l = (lf - li) / 2
    m = n
    # Constant term
    a0 = (2.0 / l) * (integrate.quad(lambda x: (x**6 * np.exp(-x**2 / 6)), 0, l))[0]
    # Cosine coefficents
    an = []
    an.append(a0 / 2.0)

    for i in range(1, n + 1):
        an_i = (2.0 / l) * (integrate.quad(lambda x: (x**6 * np.exp(-x*2 / 6)) * np.cos(i * np.pi * x / l), 0, l))[
            0]
        an.append(an_i)

    fx = a0
    s = sum(an[i] * math.cos((i * x * np.pi) / l) for i in range(1, n + 1))
    fx += s

    return an,fx

# Точка, в якій обчислюється наближення рядом Фур'є
x = math.pi / 3

an, fx = fourier(li, lf, x, n)

print('Коефіцієнти Фур\'є a_n =' + str(an))

print("Наближення рядом Фур'є з точністю до порядку", n, "в точцi x =", x, "дорівнює", fx)


# Створення масиву частот k
k_n = np.arange(0, n + 1)

# Побудова графіків гармонік та відповідних функцій an
plt.subplot(2, 1, 1)
plt.stem(k_n, an)
plt.title('Графік гармоніки an')

plt.show()

#  Завдання 3

def f(x):
    return x**6 * np.exp(-x**2 / 6)


x = np.arange(li, lf, 0.3)

y_approx = []
y_exact = []
y_approx_all = []
for i in x:
    an, fx = fourier(li, lf, i, n)
    y_approx.append(fx)


for j in range(1,n):
    for i in x:
        an_all, fx_all = fourier(li, lf, i, j)
        y_approx_all.append(fx_all)
    plt.plot(x, y_approx_all)
    y_approx_all = []


for i in x:
    y_exact.append(f(i))

# Обчислення відносної похибки наближення
erorr = []

for i in range(0, len(y_exact)):
    erorr.append((y_approx[i] - y_exact[i]) / y_exact[i])

relative_error = np.abs(erorr)

plt.plot(x, y_exact, label='Точне значення функції')
plt.plot(x, y_approx, label='Наближане значення розкладу Фур\' є')
print(y_approx_all)

plt.legend()
plt.show()

# Графік відносної похибки
plt.plot(x, relative_error)
plt.title('Відносна похибка наближення')
plt.show()

print('Відносна похибка наближення:', relative_error)


# відкриття файлу для запису
with open('output.txt', 'w') as file:
    file.write("Порядок:" + str(n))
    file.write("\nОбчислені коефіцієнти an:"+str(an))
    file.write("\nОбчислені похибки відхилень:" + str(relative_error))
    file.close()


