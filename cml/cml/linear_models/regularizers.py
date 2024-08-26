"""
>>> cml.linear_models.regularizers

Регуляризаторы, которые могут быть встроены в функции потерь
"""

import numpy as np


class BaseRegularizer:
    """
    Базовый класс для регуляризаторов, используемых в функциях потерь.

    Параметры
    ---------
    coef : float
        Коэффициент регуляризации, который масштабирует регуляризационный член.
    """

    def __init__(self, coef: float):
        self.coef_ = coef

    def __repr__(self):
        return f"{self.__class__.__name__}(coef={self.coef_})"

    def calc_reg(self, w: np.ndarray, ignore_first: bool = True) -> np.float64:
        """
        Вычисляет регуляризационный член на основе весов и коэффициента регуляризации.

        Параметры
        ---------
        w : np.ndarray
            1D массив весов модели.
        ignore_first : bool, опционально
            Если True, первый элемент массива весов считается интерсептом и не включается 
            в расчет регуляризации. По умолчанию True.

        Возвращает
        ----------
        np.float64
            Вычисленный регуляризационный член.
        """
        return 0

    def calc_grad(self, w: np.ndarray, ignore_first: bool = True) -> np.ndarray:
        """
        Вычисляет градиент регуляризационного члена по отношению к весам.

        Параметры
        ---------
        w : np.ndarray
            1D массив весов модели.
        ignore_first : bool, опционально
            Если True, расчет градиента исключает первый элемент массива весов, считая его 
            интерсептом. По умолчанию True.

        Возвращает
        ----------
        np.ndarray
            1D массив, содержащий градиент регуляризационного члена по отношению к каждому весу.
        """
        return w * 0


class L1Regularizer(BaseRegularizer):
    def calc_reg(self, w: np.ndarray, ignore_first: bool = True):
        if ignore_first:
            return self.coef_ * np.abs(w[1:]).sum()
        return self.coef_ * np.abs(w).sum()

    def calc_grad(self, w: np.ndarray, ignore_first: bool = True):
        g = self.coef_ * np.sign(w)
        if ignore_first:
            g[0] = 0
        return g


class L2Regularizer(BaseRegularizer):
    def calc_reg(self, w: np.ndarray, ignore_first: bool = True):
        if ignore_first:
            return self.coef_ * (w[1:] ** 2).sum()
        return self.coef_ * (w ** 2).sum()

    def calc_grad(self, w: np.ndarray, ignore_first: bool = True):
        g = self.coef_ * 2 * w
        if ignore_first:
            g[0] = 0
        return g
