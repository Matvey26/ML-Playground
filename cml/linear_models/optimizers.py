"""
>>> cml.linear_models.optimizers

Реализация различных оптимизаторов, которые могут быть использованы для обучения линейной модели.
"""

import numpy as np
from .losses import BaseLoss


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray,
    loss: BaseLoss,
    learning_rate: callable = lambda k: 0.001,
    n_iterations: int = 10000,
    save_path: bool = False
) -> np.ndarray:
    """
    Выполняет классическую оптимизацию методом градиентного спуска.

    Параметры
    ---------
    X : np.ndarray
        Обучающие данные, 2D массив размером (N, D), где N - размер выборки,
        а D - количество признаков. Первый столбец обычно зарезервирован для интерсепта.
    y : np.ndarray
        Целевые значения, 1D массив длины N, где N - размер выборки.
    w_init : np.ndarray
        Начальные веса, 1D массив длины D, где D - количество признаков. Первый элемент 
        обычно представляет интерсепт.
    loss : BaseLoss
        Объект функции потерь для вычисления градиента.
    learning_rate : callable, опционально
        Гиперпараметр, который определяет скорость обучения как функцию от номера итерации. 
        По умолчанию это постоянная скорость обучения 0.001.
    n_iterations : int, опционально
        Количество итераций градиентного спуска. По умолчанию 10,000.
    save_path : bool, опционально
        Если True, функция сохраняет все промежуточные веса во время обучения и возвращает 
        список этих весов. По умолчанию False.

    Возвращает
    ----------
    np.ndarray или list
        Конечные веса после оптимизации, если `save_path == Fasse`, или список 
        массивов весов для каждой итерации, если `save_path == True`.
    """
    w_prev = w_init.copy()
    path = []
    if save_path:
        path.append(w_init)

    for i in range(n_iterations):
        w_cur = w_prev - learning_rate(i) * loss.calc_grad(X, y, w_prev)
        if save_path:
            path.append(w_cur)
        w_prev = w_cur

    if save_path:
        return path
    return w_prev


def stochastic_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray,
    loss: BaseLoss,
    batch_size: int,
    learning_rate: callable = lambda k: 0.001,
    n_epoch: int = 100,
    save_path: bool = False
) -> np.ndarray:
    """
    Выполняет оптимизацию методом стохастического градиентного спуска (SGD).

    Параметры
    ---------
    X : np.ndarray
        Обучающие данные, 2D массив размером (N, D), где N - размер выборки,
        а D - количество признаков. Первый столбец обычно зарезервирован для интерсепта.
    y : np.ndarray
        Целевые значения, 1D массив длины N, где N - количество выборок.
    w_init : np.ndarray
        Начальные веса, 1D массив длины D, где D - количество признаков. Первый элемент 
        обычно представляет интерсепт.
    loss : BaseLoss
        Объект функции потерь для вычисления градиента.
    batch_size : int
        Размер мини-батча, используемого для оценки градиента.
    learning_rate : callable, опционально
        Гиперпараметр, который определяет скорость обучения как функцию номера итерации. 
        По умолчанию это постоянная скорость обучения 0.001.
    n_epoch : int, опционально
        Количество эпох обучения, где одна эпоха означает полный проход по всему набору данных. 
        По умолчанию 100.
    save_path : bool, опционально
        Если True, функция сохраняет все промежуточные веса во время обучения и возвращает 
        список этих весов. По умолчанию False.

    Возвращает
    ----------
    np.ndarray или list
        Конечные веса после оптимизации, если `save_path == False`, или список 
        массивов весов для каждой эпохи, если `save_path == True`.
    """
    shuffled_indices = np.arange(X.shape[0])
    np.random.shuffle(shuffled_indices)

    Xs = X.copy()[shuffled_indices]
    ys = y.copy()[shuffled_indices]

    w_prev = w_init.copy()
    path = []
    if save_path:
        path.append(w_init)

    k = 0
    for _ in range(n_epoch):
        for i in range(0, X.shape[0], batch_size):
            X_batch = Xs[i:i+batch_size]
            y_batch = ys[i:i+batch_size]
            w_cur = w_prev - learning_rate(k) * loss.calc_grad(X_batch, y_batch, w_prev)
            if save_path:
                path.append(w_cur)
            w_prev = w_cur

            k += 1

    if save_path:
        return path
    return w_prev
