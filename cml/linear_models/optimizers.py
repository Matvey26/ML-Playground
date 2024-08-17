"""
>>> cml.linear_models.optimizers

The implementation of various optimizers that can be used to train a linear model.
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
    Performs classic gradient descent optimization.

    Parameters
    ----------
    X : np.ndarray
        Training data, a 2D array with shape (N, D + 1), where N is the number of samples
        and D is the number of features. The first column is typically
        reserved for the intercept.
    y : np.ndarray
        Target values, a 1D array of length N, where N is the number of samples.
    w_init : np.ndarray
        Initial weights, a 1D array of length D, where D is the number of features. The first 
        element is typically the intercept.
    loss : BaseLoss
        Loss function object to compute the gradient.
    learning_rate : callable, optional
        Hyperparameter that defines the learning rate as a function of the iteration number. 
        Default is a constant learning rate of 0.001.
    n_iterations : int, optional
        Number of gradient descent iterations. Default is 10,000.
    save_path : bool, optional
        If True, the function saves all intermediate weights during training and returns 
        a list of these weights. Default is False.

    Returns
    -------
    np.ndarray or list
        The final weights after optimization if `save_path` is False, or a list of weight 
        arrays from each iteration if `save_path` is True.
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
    batch_size: int = 100,
    learning_rate: callable = lambda k: 0.001,
    n_epoch: int = 100,
    save_path: bool = False
) -> np.ndarray:
    """
    Performs stochastic gradient descent (SGD) optimization.

    Parameters
    ----------
    X : np.ndarray
        Training data, a 2D array with shape (N, D), where N is the number of samples and 
        D is the number of features. The first column is typically reserved for the intercept.
    y : np.ndarray
        Target values, a 1D array of length N, where N is the number of samples.
    w_init : np.ndarray
        Initial weights, a 1D array of length D, where D is the number of features. The first 
        element is typically the intercept.
    loss : BaseLoss
        Loss function object to compute the gradient.
    batch_size : int, optional
        Size of the mini-batch used to estimate the gradient. Default is 100.
    learning_rate : callable, optional
        Hyperparameter that defines the learning rate as a function of the iteration number. 
        Default is a constant learning rate of 0.001.
    n_epoch : int, optional
        Number of epochs for training, where one epoch means one full pass over the dataset. 
        Default is 100.
    save_path : bool, optional
        If True, the function saves all intermediate weights during training and returns 
        a list of these weights. Default is False.

    Returns
    -------
    np.ndarray or list
        The final weights after optimization if `save_path` is False, or a list of weight 
        arrays from each epoch if `save_path` is True.
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
