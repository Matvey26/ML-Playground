"""
>>> cml.linear_models.losses

Функции потерь с возможностью добавления регуляризации.
Используются в основном для обучения моделей, а не для оценки метрик.
"""

import numpy as np
from .regularizers import BaseRegularizer


class BaseLoss:
    """
    Базовый класс для функций потерь с возможностью регуляризации.

    Параметры
    ---------
    regularizer : BaseRegularizer, опционально
        Объект регуляризатора, который используется для расчета функции потерь. 
        По умолчанию BaseRegularizer(0).
    """

    def __init__(self, regularizer: BaseRegularizer = BaseRegularizer(0)):
        self.regularizer = regularizer

    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.float64:
        """
        Вычисляет значение функции потерь с учетом регуляризации.

        Параметры
        ---------
        X : np.ndarray
            Матрица признаков, 2D массив размером (N, D), где N — размер выборки,
            а D — количество признаков. Обычно первый столбец X[:, 0] отвечает за константу (интерсепт).
        y : np.ndarray
            Вектор ответов (целевых значений), 1D массив длины N, где N — размер выборки.
        w : np.ndarray
            Вектор весов модели, 1D массив длины D, где D — количество признаков.
            Первый элемент w[0] обычно представляет собой интерсепт.

        Возвращает
        ----------
        np.float64
            Значение функции потерь с учетом регуляризации.
        """
        return 0

    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Вычисляет градиент функции потерь по отношению к весам модели с учетом регуляризации.

        Параметры
        ---------
        X : np.ndarray
            Матрица признаков, 2D массив размером (N, D), где N — размер выборки,
            а D — количество признаков. Обычно первый столбец X[:, 0] отвечает за константу (интерсепт).
        y : np.ndarray
            Вектор ответов (целевых значений), 1D массив длины N, где N — размер выборки.
        w : np.ndarray
            Вектор весов модели, 1D массив длины D, где D — количество признаков.
            Первый элемент w[0] обычно представляет собой интерсепт.

        Возвращает
        ----------
        np.ndarray
            Градиент функции потерь по отношению к весам, 1D массив длины D.
        """
        return w * 0


class MSELoss(BaseLoss):
    """
    Класс для функции потерь MSE (среднеквадратичная ошибка) с возможностью регуляризации.

    Параметры
    ---------
    regularizer : BaseRegularizer, опционально
        Объект регуляризатора, который используется для расчета функции потерь. 
        По умолчанию используется экземпляр BaseRegularizer с нулевым коэффициентом регуляризации.
    """

    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.float64:
        Q = np.mean((np.dot(X, w) - y) ** 2)
        R = self.regularizer.calc_reg(w)

        return Q + R

    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        err = np.dot(X, w) - y
        Q = 2 * np.dot(X.T, err) / len(y)
        R = self.regularizer.calc_grad(w)

        return Q + R


class MAELoss(BaseLoss):
    """
    Класс для функции потерь MAE (средняя абсолютная ошибка) с возможностью регуляризации.

    Параметры
    ---------
    regularizer : BaseRegularizer, опционально
        Объект регуляризатора, который используется для расчета функции потерь. 
        По умолчанию используется экземпляр BaseRegularizer с нулевым коэффициентом регуляризации.
    """

    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.float64:
        Q = np.mean(np.abs(np.dot(X, w) - y))
        R = self.regularizer.calc_reg(w)

        return Q + R

    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        err = np.dot(X, w) - y
        Q = np.dot(X.T, np.sign(err)) / len(y)
        R = self.regularizer.calc_grad(w)

        return Q + R


class HuberLoss(BaseLoss):
    """
    Класс для функции потерь Хьюбера (Huber) с возможностью регуляризации.

    Параметры
    ---------
    delta : float
        Пороговое значение для функции потерь Хьюбера.
    regularizer : BaseRegularizer, опционально
        Объект регуляризатора, который используется для расчета функции потерь. 
        По умолчанию BaseRegularizer(0).
    """

    def __init__(self, delta: float, regularizer: BaseRegularizer = BaseRegularizer(0)):
        super().__init__(regularizer)
        self.delta = delta

    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.float64:
        z = np.dot(X, w) - y
        phi = np.where(
            np.abs(z) < self.delta,
            0.5 * z ** 2,
            self.delta * (np.abs(z) - 0.5 * self.delta)
        )
        Q = phi.mean()
        R = self.regularizer.calc_reg(w)

        return Q + R

    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        z = np.dot(X, w) - y
        phi = np.where(
            np.abs(z) < self.delta,
            z,
            self.delta * np.sign(z)
        )
        Q = np.dot(X.T, phi) / len(y)
        R = self.regularizer.calc_grad(w)

        return Q + R
