import numpy as np
import matplotlib.pyplot as plt

# Задание данных
N = 1000
x = np.linspace(0, 1, N)
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

# Функция для полиномиальной регрессии
def polynomial_regression(x, t, M):
    X = np.vstack([x**i for i in range(M + 1)]).T  # Матрица признаков
    w = np.linalg.pinv(X) @ t                      # Аналитическое решение
    y = X @ w                                      # Предсказания модели
    mse = np.mean((y - z)**2)                      # Среднеквадратичная ошибка
    return y, w, mse

# Построение графиков для M = 1, 8, 100
Ms = [1, 8, 100]
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

for i, M in enumerate(Ms):
    y, w, _ = polynomial_regression(x, t, M)
    axs[i].plot(x, z, label='z(x)', color='blue')
    axs[i].scatter(x, t, label='t(x)', color='red', s=5, alpha=0.5)
    axs[i].plot(x, y, label=f'Regression (M={M})', color='green')
    axs[i].legend()
    axs[i].set_title(f'Полиномиальная регрессия при M={M}')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('y')

plt.tight_layout()
plt.show()

# График зависимости ошибки от степени полинома M
errors = []
M_values = range(1, 101)

for M in M_values:
    _, _, mse = polynomial_regression(x, t, M)
    errors.append(mse)

plt.figure(figsize=(10, 5))
plt.plot(M_values, errors, marker='o', markersize=3)
plt.title('Ошибка E(w) в зависимости от степени полинома M')
plt.xlabel('Степень полинома M')
plt.ylabel('Среднеквадратичная ошибка')
plt.grid(True)
plt.tight_layout()
plt.show()
