import cml.linear_models.regularizers as regularizers
import numpy as np
import pytest


@pytest.fixture
def init_weights():
    n = 3
    return np.arange(-n, n + 1)


# Тест L1 регуляризатора при подсчёте лосса
@pytest.mark.parametrize("ignore_first, expected_reg1, expected_reg2", [
    (True, 9, 18),
    (False, 12, 24)
])
def test_l1_reg(init_weights, ignore_first, expected_reg1, expected_reg2):
    l1_reg_1 = regularizers.L1Regularizer(1)
    l1_reg_2 = regularizers.L1Regularizer(2)
    w = init_weights

    assert l1_reg_1.calc_reg(w, ignore_first=ignore_first) == expected_reg1
    assert l1_reg_2.calc_reg(w, ignore_first=ignore_first) == expected_reg2


# Тест L1 регуляризатора при подсчёте градиента
@pytest.mark.parametrize("ignore_first, expected_grad1, expected_grad2", [
    (True, np.array([0, -1, -1, 0, 1, 1, 1]), np.array([0, -2, -2, 0, 2, 2, 2])),
    (False, np.array([-1, -1, -1, 0, 1, 1, 1]), np.array([-2, -2, -2, 0, 2, 2, 2]))
])
def test_l1_grad(init_weights, ignore_first, expected_grad1, expected_grad2):
    l1_reg_1 = regularizers.L1Regularizer(1)
    l1_reg_2 = regularizers.L1Regularizer(2)
    w = init_weights

    assert np.array_equal(l1_reg_1.calc_grad(w, ignore_first=ignore_first), expected_grad1)
    assert np.array_equal(l1_reg_2.calc_grad(w, ignore_first=ignore_first), expected_grad2)


# Тест L2 регуляризатора при подсчёте лосса
@pytest.mark.parametrize("ignore_first, expected_reg1, expected_reg2", [
    (True, 19, 38),
    (False, 28, 56)
])
def test_l2_reg(init_weights, ignore_first, expected_reg1, expected_reg2):
    l2_reg_1 = regularizers.L2Regularizer(1)
    l2_reg_2 = regularizers.L2Regularizer(2)
    w = init_weights

    assert l2_reg_1.calc_reg(w, ignore_first=ignore_first) == expected_reg1
    assert l2_reg_2.calc_reg(w, ignore_first=ignore_first) == expected_reg2


# Тест L2 регуляризатора при подсчёте градиента
@pytest.mark.parametrize("ignore_first, expected_grad1, expected_grad2", [
    (True, np.array([0, -4, -2, 0, 2, 4, 6]), np.array([0, -8, -4, 0, 4, 8, 12])),
    (False, np.array([-6, -4, -2, 0, 2, 4, 6]), np.array([-12, -8, -4, 0, 4, 8, 12]))
])
def test_l2_grad(init_weights, ignore_first, expected_grad1, expected_grad2):
    l2_reg_1 = regularizers.L2Regularizer(1)
    l2_reg_2 = regularizers.L2Regularizer(2)
    w = init_weights

    assert np.array_equal(l2_reg_1.calc_grad(w, ignore_first=ignore_first), expected_grad1)
    assert np.array_equal(l2_reg_2.calc_grad(w, ignore_first=ignore_first), expected_grad2)
