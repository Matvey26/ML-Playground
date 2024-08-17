"""
>>> cml.linear_models.regularizers

Regularizers that can be integrated into your loss functions
"""

import numpy as np


class BaseRegularizer:
    """
    Base class for regularizers used in loss functions.

    Parameters
    ----------
    coef : float
        Regularization coefficient that scales the regularization term.
    """

    def __init__(self, coef: float):
        self.coef_ = coef

    def calc_reg(self, w: np.ndarray, ignore_first: bool = True) -> np.float64:
        """
        Computes the regularization term based on the weights and the regularization coefficient.

        Parameters
        ----------
        w : np.ndarray
            A 1D array of model weights.
        ignore_first : bool, optional
            If True, the first element of the weights array is considered as an intercept and is not included 
            in the regularization calculation. Default is True.

        Returns
        -------
        np.float64
            The computed regularization term.
        """
        return 0

    def calc_grad(self, w: np.ndarray, ignore_first: bool = True) -> np.ndarray:
        """
        Calculates the gradient of the regularization term with respect to the weights.

        Parameters
        ----------
        w : np.ndarray
            A 1D array of model weights.
        ignore_first : bool, optional
            If True, the gradient calculation excludes the first element of the weights array, assuming it is 
            an intercept. Default is True.

        Returns
        -------
        np.ndarray
            A 1D array containing the gradient of the regularization term with respect to each weight.
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
