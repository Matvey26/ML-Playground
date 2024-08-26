"""
>>> cml.linear_models.optimizers

Реализация различных оптимизаторов, которые могут быть использованы для обучения линейной модели.
"""

import numpy as np
from .losses import BaseLoss
from typing import Callable, Any


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray,
    loss: BaseLoss,
    learning_rate: Callable[[int], float] = lambda k: 0.001,
    stop_function: Callable[[np.ndarray, int], bool] = lambda w, k: k < 10_000,
    callback: Callable[[np.ndarray, int], Any] = lambda w, k: None

) -> np.ndarray:
    """
    Выполняет классическую оптимизацию методом градиентного спуска.

    Параметры
    ---------
    `X : np.ndarray`
        Обучающие данные, 2D массив размером (N, D), где N - размер выборки,
        а D - количество признаков. Первый столбец обычно зарезервирован для интерсепта.
    `y : np.ndarray`
        Целевые значения, 1D массив длины N, где N - размер выборки.
    `w_init : np.ndarray`
        Начальные веса, 1D массив длины D, где D - количество признаков. Первый элемент 
        обычно представляет интерсепт.
    `loss : BaseLoss`
        Объект функции потерь для вычисления градиента.
    `learning_rate : Callable[[int], float]`, опционально
        Гиперпараметр, который определяет скорость обучения как функцию от номера итерации. 
        По умолчанию это постоянная скорость обучения 0.001.
    `stop_function : Callable[[np.ndarray, int], bool]`, опционально
        Правило остановки градиентного спуска, функция, которая принимает на вход веса и
        номер итерации и возвращает булево значение. По умолчанию `lambda w, k: k < 10_000`
    `callback: Callable[[np.ndarray, int], Any]`, опционально
        Функция, которая вызывается на каждой итерации градиентного спуска, принимает веса
        и номер итерации.

    Возвращает
    ----------
    np.ndarray
        Конечные веса после оптимизации.
    """
    w = w_init.copy()
    callback(w, -1)
    
    i = 0
    while stop_function(w, i):
        w -= learning_rate(i) * loss.calc_grad(X, y, w)
        callback(w, i)
        i += 1

    return w


def stochastic_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray,
    loss: BaseLoss,
    batch_size: int,
    learning_rate: Callable[[int], float] = lambda k: 0.001,
    stop_function: Callable[[np.ndarray, int], bool] = lambda w, k: k < 10_000,
    callback: Callable[[np.ndarray, int], Any] = lambda w, k: None
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
    learning_rate : Callable[[int], float], опционально
        Гиперпараметр, который определяет скорость обучения как функцию номера итерации. 
        По умолчанию это постоянная скорость обучения 0.001.
    stop_function : Callable[[np.ndarray, int], bool], опционально
        Правило остановки SGD, функция, которая принимает на вход веса и номер итерации
        и возвращает булево значение. По умолчанию `lambda w, k: k < 10_000`
    callback : Callable[[np.ndarray, int], Any], опционально
        Функция, которая вызывается на каждой итерации SGD, принимает веса и номер итерации.

    Возвращает
    ----------
    np.ndarray
        Конечные веса после оптимизации.
    """
    shuffled_indices = np.arange(X.shape[0])
    np.random.shuffle(shuffled_indices)

    Xs = X[shuffled_indices]
    ys = y[shuffled_indices]

    w = w_init.copy()
    callback(w, -1)
    
    k = 0
    while stop_function(w, k):
        for i in range(0, X.shape[0], batch_size):
            X_batch = Xs[i:i+batch_size]
            y_batch = ys[i:i+batch_size]
            
            w -= learning_rate(k) * loss.calc_grad(X_batch, y_batch, w)
            callback(w, k)
            k += 1

            if not stop_function(w, k):
                break

    return w
