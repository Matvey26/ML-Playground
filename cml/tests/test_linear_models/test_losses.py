import cml.linear_models.losses as losses
import cml.linear_models.regularizers as regularizers
import pytest
import numpy as np


@pytest.fixture
def init_X_y_w():
    X = np.array([[1, 1, 2], [1, 3, 4], [1, 0, 6], [1, -1, -1]])
    y = np.array([1, 2, 3, 4])
    w = np.array([1, -1, 2])

    return X, y, w


@pytest.mark.parametrize('is_first_intercept, regularizer, expected_value', [
    (True, regularizers.BaseRegularizer(0), 5.25),     # Пример без регуляризации, не игнорируя первый признак
    (False, regularizers.BaseRegularizer(0), 4.75),  # Пример без регуляризации, игнорируя первый признак
    (True, regularizers.L2Regularizer(0.1), 5.75),  # Пример с L2 регуляризацией, не игнорируя первый признак
    (False, regularizers.L2Regularizer(0.1), 5.25), # Пример с L2 регуляризацией, игнорируя первый признак
])
def test_mae_loss(init_X_y_w, is_first_intercept: bool, regularizer: regularizers.BaseRegularizer, expected_value: int):
    mae = losses.MAELoss(regularizer)
    X, y, w = init_X_y_w
    if not is_first_intercept:
        X = X[:, 1:]
        w = w[1:]

    assert abs(mae.calc_loss(X, y, w, is_first_intercept) - expected_value) < 10 ** (-10)


@pytest.mark.parametrize('is_first_intercept, regularizer, expected_value', [
    (True, regularizers.BaseRegularizer(0), 35.25),            # Пример без регуляризации, не игнорируя первый признак
    (False, regularizers.BaseRegularizer(0), 29.75),        # Пример без регуляризации, игнорируя первый признак
    (True, regularizers.L2Regularizer(0.1), 35.75),     # Пример с L2 регуляризацией, не игнорируя первый признак
    (False, regularizers.L2Regularizer(0.1), 30.25), # Пример с L2 регуляризацией, игнорируя первый признак
])
def test_mse_loss(init_X_y_w, is_first_intercept: bool, regularizer: regularizers.BaseRegularizer, expected_value: int):
    mse = losses.MSELoss(regularizer)
    X, y, w = init_X_y_w
    if not is_first_intercept:
        X = X[:, 1:]
        w = w[1:]

    assert abs(mse.calc_loss(X, y, w, is_first_intercept) - expected_value) < 10 ** (-10)


@pytest.mark.parametrize('is_first_intercept, regularizer, expected_value', [
    (True, regularizers.BaseRegularizer(0), 13.125),     # Пример без регуляризации, не игнорируя первый признак
    (False, regularizers.BaseRegularizer(0), 11.625),  # Пример без регуляризации, игнорируя первый признак
    (True, regularizers.L2Regularizer(0.1), 13.625),  # Пример с L2 регуляризацией, не игнорируя первый признак
    (False, regularizers.L2Regularizer(0.1), 12.125), # Пример с L2 регуляризацией, игнорируя первый признак
])
def test_huber_loss(init_X_y_w, is_first_intercept: bool, regularizer: regularizers.BaseRegularizer, expected_value: int):
    huber = losses.HuberLoss(4, regularizer)
    X, y, w = init_X_y_w
    if not is_first_intercept:
        X = X[:, 1:]
        w = w[1:]

    assert abs(huber.calc_loss(X, y, w, is_first_intercept) - expected_value) < 10 ** (-10)
