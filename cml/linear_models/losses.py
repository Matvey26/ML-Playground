"""
>>> cml.linear_models.losses

Loss functions in which a regularizer can be embedded.
They are used more for training models than for calculating metrics.
"""

import numpy as np
from .regularizers import BaseRegularizer


class BaseLoss:
    """
    Base class for loss functions with an optional regularizer.

    Parameters
    ----------
    regularizer : BaseRegularizer, optional
        Regularizer object to be used in the loss calculation. Default is an instance of 
        BaseRegularizer with a regularization strength of 0.
    """

    def __init__(self, regularizer: BaseRegularizer = BaseRegularizer(0)):
        self.regularizer = regularizer

    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.float64:
        """
        Calculates the loss value for a given dataset, targets, and model weights, 
        incorporating the regularization term if provided.

        Parameters
        ----------
        X : np.ndarray
            Training sample, a 2D array with a shape of (N, D + 1), where N is the number 
            of samples and D is the number of features. The first column X[:, 0] 
            is typically reserved for the intercept (constant feature).
        y : np.ndarray
            Target values, a 1D array of length N, where N is the number of samples.
        w : np.ndarray
            Weights of the linear model, a 1D array of length D, where D is the number 
            of features. The first element w[0] typically represents the intercept.

        Returns
        -------
        np.float64
            The computed loss value, including the regularization term if applicable.
        """
        return 0

    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Calculates the gradient of the loss function with respect to the model weights, 
        incorporating the gradient of the regularization term if provided.

        Parameters
        ----------
        X : np.ndarray
            Training sample, a 2D array with a shape of (N, D + 1), where N is the number 
            of samples and D is the number of features. The first column X[:, 0] 
            is typically reserved for the intercept (constant feature).
        y : np.ndarray
            Target values, a 1D array of length N, where N is the number of samples.
        w : np.ndarray
            Weights of the linear model, a 1D array of length D, where D is the number 
            of features. The first element w[0] typically represents the intercept.

        Returns
        -------
        np.ndarray
            The gradient of the loss function with respect to the weights, a 1D array 
            of length D.
        """
        return w * 0


class MSELoss(BaseLoss):
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
    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        Q = np.mean(np.abs(np.dot(X, w) - y))
        R = self.regularizer.calc_reg(w)

        return Q + R

    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        err = np.dot(X, w) - y
        Q = np.dot(X.T, np.sign(err)) / len(y)
        R = self.regularizer.calc_grad(w)

        return Q + R
