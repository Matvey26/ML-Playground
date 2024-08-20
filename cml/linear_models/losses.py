"""
>>> cml.linear_models.losses

Функции потерь, в которые можно встроить регуляризатор.
Они используются больше для обучения моделей, чем для расчета метрик.
"""

import numpy as np
from .regularizers import BaseRegularizer


class BaseLoss:
    """
    Базовый класс для функций потерь с опциональным регуляризатором.

    Параметры
    ---------
    regularizer : BaseRegularizer, опционально
        Объект регуляризатора, который будет использоваться в расчете функции потерь. 
        По умолчанию используется экземпляр BaseRegularizer с коэффициентом регуляризации 0.
    """

    def __init__(self, regularizer: BaseRegularizer = BaseRegularizer(0)):
        self.regularizer = regularizer

    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.float64:
        """
        Вычисляет значение функции потерь для заданного набора данных, целевых значений и весов модели,
        включая регуляризационный член, если он предоставлен.

        Параметры
        ---------
        X : np.ndarray
            Обучающая выборка, 2D массив размером (N, D), где N - размер выборки,
            а D - количество признаков. Первый столбец X[:, 0] обычно зарезервирован для интерсепта (постоянный признак).
        y : np.ndarray
            Целевые значения, 1D массив длины N, где N - размер выборки.
        w : np.ndarray
            Веса линейной модели, 1D массив длины D, где D - количество признаков.
            Первый элемент w[0] обычно представляет интерсепт.

        Возвращает
        ----------
        np.float64
            Вычисленное значение функции потерь, включая регуляризационный член, если он указан.
        """
        return 0

    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Вычисляет градиент функции потерь по отношению к весам модели,
        включая градиент регуляризационного члена, если он предоставлен.

        Параметры
        ---------
        X : np.ndarray
            Обучающая выборка, 2D массив размером (N, D), где N - размер выборки,
            а D - количество признаков. Первый столбец X[:, 0] обычно зарезервирован для интерсепта (постоянный признак).
        y : np.ndarray
            Целевые значения, 1D массив длины N, где N - размер выборки.
        w : np.ndarray
            Веса линейной модели, 1D массив длины D, где D - количество признаков.
            Первый элемент w[0] обычно представляет интерсепт.

        Возвращает
        ----------
        np.ndarray
            Градиент функции потерь по отношению к весам, 1D массив длины D.
        """
        return w * 0


class MSELoss(BaseLoss):
    """
    Класс для функции потерь MSE с опциональным регуляризатором

    Параметры
    ---------
    regularizer : BaseRegularizer, опционально
        Объект регуляризатора, который будет использоваться в расчете функции потерь. 
        По умолчанию используется экземпляр BaseRegularizer с коэффициентом регуляризации 0.
    """

    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        Q = np.mean((np.dot(X, w) - y) ** 2)
        R = self.regularizer.calc_reg(w)

        return Q + R

    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        err = np.dot(X, w) - y
        Q = 2 * np.dot(X.T, err) / len(y)
        R = self.regularizer.calc_grad(w)

        return Q + R


class MAELoss(BaseLoss):
    """
    Класс для функции потерь MAE с опциональным регуляризатором

    Параметры
    ---------
    regularizer : BaseRegularizer, опционально
        Объект регуляризатора, который будет использоваться в расчете функции потерь. 
        По умолчанию используется экземпляр BaseRegularizer с коэффициентом регуляризации 0.
    """

    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        Q = np.mean(np.abs(np.dot(X, w) - y))
        R = self.regularizer.calc_reg(w)

        return Q + R

    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        err = np.dot(X, w) - y
        Q = np.dot(X.T, np.sign(err)) / len(y)
        R = self.regularizer.calc_grad(w)

        return Q + R


class HuberLoss(BaseLoss):
    """
    Класс для функции потерь Huber с коэффициентом дельта и опциональным регуляризатором

    Параметры
    ---------
    delta : float
        Коэффициент функции потерь Хьюбера.
    regularizer : BaseRegularizer, опционально
        Объект регуляризатора, который будет использоваться в расчете функции потерь. 
        По умолчанию используется экземпляр BaseRegularizer с коэффициентом регуляризации 0.
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
