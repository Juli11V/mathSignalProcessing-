import numpy as np
import matplotlib.pyplot as plt
import time

def coeff_k(k, x):
    N = len(x)
    n = np.arange(N)
    A_k = np.sum(x * np.cos(2 * np.pi * k * n / N)) / N
    B_k = np.sum(x * np.sin(2 * np.pi * k * n / N)) / (-N)
    C_k = A_k + 1j * B_k
    return C_k

def phase_amp(x):
    N = len(x)
    amp = np.zeros(N)
    phase = np.zeros(N)
    for k in range(N):
        coeff = coeff_k(k, x)
        A_k = coeff.real
        B_k = coeff.imag
        amp[k] = np.sqrt(A_k**2 + B_k**2)
        phase[k] = np.arctan2(B_k, A_k)
    return amp, phase

N = 16
x = np.random.rand(N)

#Обчисленяя коефіцієнтів ряду Фур’є та амплітудний і фазовий спектри
start = time.time()

coeff = []
for k in range(N):
    C_k = coeff_k(k, x)
    print(f"C_{k} = [ {C_k:.8f} ]")
    coeff.append(coeff_k(k, x))
coeff = np.array(coeff)
amp, phase = phase_amp(x)

end = time.time()


print(f"\nЧас виконання обрахунків у програмі - {end - start:.8f} секунд")

num_operations = N*((5*N+5*N+2)+1)
print(f"Кількість операцій (додавання, множення) - {num_operations}")


fig, axs = plt.subplots(2, 1, figsize=(8, 8))
amp = amp[1:]
axs[0].stem(amp)
axs[0].set_title('Графік спектру амплітуд')
axs[0].set_xlabel('k')
axs[0].set_ylabel('Амплітуда')
axs[1].stem(phase)
axs[1].set_title('Графік спектру фаз')
axs[1].set_xlabel('k')
axs[1].set_ylabel('Фази')
plt.tight_layout()
plt.show()


