import streamlit as st
import numpy as np
import plotly.graph_objs as go
from typing import Callable, Tuple

# Нужные функции


def real_dependence(x: np.ndarray) -> np.ndarray:
    """Реальная зависимость y от x.

    Параметры
    ---------
    x : np.ndarray
        Одномерный массив чисел.

    Возвращает
    ----------
    y : np.ndarray
        Результат некоторой функции от массива x.
    """
    return np.sin(x)


def make_dataset(
    N: int,
    xlim: Tuple[float, float],
    f: Callable[[np.ndarray], np.ndarray],
    noise_x: float = 0.0,
    noise_y: float = 0.0,
    outliers: float = 0.0,
    outlier_lim: Tuple[float, float] = (-5, 5)
) -> Tuple[np.ndarray, np.ndarray]:
    """Генерирует датасет с одним признаком с учётом указанного уровня шума и количества выбросов.

    Параметры
    ---------
    N : int
        Желаемое число объектов в выборке
    xlim : Tuple[float, float]
        Кортеж из двух значений, определяющий пределы x
    f : function
        Реальная зависимость между признаком и таргетом
    noise_x : float
        Количество шума для X. `noise_x = 0` означает отсутствие шума.
    noise_y : float
        Количество шума для y. `noise_y = 0` означает отсутствие шума.
    outliers : float
        Доля выбросов от 0 до 1.
    outlier_lim : Tuple[float, float]
        В каких пределах будут выбросы

    Возвращает
    ----------
    X : np.ndarray
        Двумерный массив, где `X.shape == (N, 1)`
    y : np.ndarray
        Одномерный массив, таргет
    """
    def add_noise(data: np.ndarray, noise: float) -> np.ndarray:
        return data + (np.random.rand(len(data)) * 2 - 1) * noise

    def make_x(n: int) -> np.ndarray:
        return np.random.uniform(*xlim, n)

    x = make_x(N)
    y = f(x)

    x = add_noise(x, noise_x)
    y = add_noise(y, noise_y)

    if outliers > 0:
        num_outliers = round(N * outliers)
        outlier_indices = np.random.choice(N, num_outliers, replace=False)
        y[outlier_indices] = np.random.uniform(*outlier_lim, num_outliers)

    return x.reshape((N, 1)), y


# Настройки приложения

with st.expander('Настроить датасет'):
    left_column, right_column = st.columns(2)

    with left_column:
        N = st.slider('Размер обучающей выборки', 50, 1000, step=50, value=150)
        noise_x = st.slider('Количество шума для x', min_value=0.0, max_value=3.0, step=0.1, value=0.2)
        outliers = st.slider('Процент выбросов', min_value=0., max_value=1., step=0.01, value=0.03)

    with right_column:
        x_lim = st.slider('В каких пределах будет x', -10, 10, (-2, 2))
        noise_y = st.slider('Количество шума для y', min_value=0.0, max_value=3.0, step=0.1, value=0.2)
        outliers_lim = st.slider('В каких пределах будут выбросы', -10, 10, (-2, 2))

X, y = make_dataset(N, x_lim, real_dependence, noise_x, noise_y, outliers, outliers_lim)

plot = st.empty()
fig = go.Figure()
fig.add_trace(go.Scatter(x=X[:, 0], y=y, mode='markers', opacity=0.4, line={'color': 'rgb(200, 200, 200)'}, name='Обучающая выборка'))
x_real = np.linspace(*x_lim, num=200)
y_real = real_dependence(x_real)
fig.add_trace(go.Scatter(x=x_real, y=y_real, mode='lines', opacity=0.7, line={'dash': 'dash', 'color': 'orange'}, name='Реальная зависимость'))
fig.update_layout(width=600, height=500)

plot.plotly_chart(fig, use_container_width=True)
