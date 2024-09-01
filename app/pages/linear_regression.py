from cml.linear_models import losses, optimizers, regularizers
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
    outlier_lim: Tuple[float, float] = (-5, 5),
    n_polinom: int = 1
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
    n_polinomial : int
        Количество полиномиальных признаков. Иначе говоря, до какой степени брать `x`.

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

    X = np.hstack([np.ones(N)] + [x ** i for i in range(1, n_polinom + 1)]).reshape((N, n_polinom + 1), order='F')
    return X, y


# Приложение

# Настройка датасета
with st.expander('Настроить датасет'):
    left_column, right_column = st.columns(2)

    with left_column:
        N = st.slider('Размер обучающей выборки', 50, 1000, step=50, value=150)
        noise_x = st.slider('Количество шума для x', min_value=0.0, max_value=3.0, step=0.1, value=0.2)
        outliers = st.slider('Процент выбросов', min_value=0., max_value=1., step=0.01, value=0.03)
        n_polinom = st.slider('Количество полиномиальных признаков', min_value=1, max_value=10, step=1, value=1)

    with right_column:
        x_lim = st.slider('В каких пределах будет x', -10, 10, (-2, 2))
        noise_y = st.slider('Количество шума для y', min_value=0.0, max_value=3.0, step=0.1, value=0.2)
        outliers_lim = st.slider('В каких пределах будут выбросы', -10, 10, (-2, 2))

X, y = make_dataset(N, x_lim, real_dependence, noise_x, noise_y, outliers, outliers_lim, n_polinom)

# Настройка модели
with st.expander('Настроить модель'):
    left_column, right_column = st.columns(2)

    with left_column:
        optimizer_name = st.selectbox(
            'Оптимизирующий метод',
            optimizers.optimizers.keys()
        )
        optimizer = optimizers.optimizers[optimizer_name]
        optimizer_kw = {}
        if optimizer_name == 'stochastic_gradient_descent':
            optimizer_kw['batch_size'] = st.slider(
                'batch size',
                min_value=1,
                max_value=N,
                step=10,
                value=1
            )

        loss_name = st.selectbox(
            'Функционал ошибки',
            losses.losses.keys()
        )
        loss = losses.losses[loss_name]
        loss_kw = {}
        if loss_name == 'Huber':
            loss_kw['delta'] = 10 ** st.slider(
                'log - delta (huber)',
                min_value=-5.0,
                max_value=5.0,
                value=0.0,
                step=0.1
            )

        regularizer_name = st.selectbox(
            'Регуляризация',
            regularizers.regularizers.keys()
        )
        regularizer = regularizers.regularizers[regularizer_name]

        reg_coef = st.slider('Коэффициент регуляризации', 0.001, 10.0, step=0.001, format='%.3f')

    with right_column:
        # st.latex(r'Шаг\ обучения\ \eta_k=\lambda \cdot \left(\frac{s_0}{s_0+k}\right)^p')
        st.write('learning_rate(k)=lambda * (s / (s + k)) ^ p')
        lmb = st.slider('lambda', 0.00001, 0.01, 0.005, step=0.00001, format='%.4f')
        s = st.slider('s', 0.01, 100.0, 50.0)
        p = st.slider('p', 0.0, 2.0, 0.4)

        n_iterations = st.slider('Количество итераций обучения', 100, 10000, step=50)

x_real = np.linspace(*x_lim, num=200)
y_real = real_dependence(x_real)

left_column, right_column = st.columns(2)
is_start_learning = left_column.button('Начать обучение')
right_column.button('Очистить данные')

plot = st.empty()
fig = go.Figure()


def callback(w: np.ndarray, k: int):
    fig.data = []

    fig.add_trace(go.Scatter(
        x=X[:, 1],
        y=y,
        mode='markers',
        opacity=0.4,
        line={'color': 'rgb(200, 200, 200)'},
        name='Обучающая выборка'
    ))

    fig.add_trace(go.Scatter(
        x=x_real,
        y=y_real,
        mode='lines',
        opacity=0.7,
        line={'dash': 'dash', 'color': 'orange'},
        name='Реальная зависимость'
    ))

    x_pred = np.hstack([np.ones(200)] + [x_real ** i for i in range(1, n_polinom + 1)]).reshape((200, n_polinom + 1), order='F')
    y_pred = np.dot(x_pred, w)
    fig.add_trace(go.Scatter(
        x=x_pred[:, 1],
        y=y_pred,
        mode='lines',
        opacity=0.7,
        line={'color': 'white'},
        name='Предсказание модели'
    ))

    fig.update_layout(width=600, height=500, title=f"Итерация #{k}")

    plot.plotly_chart(fig, use_container_width=True)


fig.add_trace(go.Scatter(
    x=X[:, 1],
    y=y,
    mode='markers',
    opacity=0.4,
    line={'color': 'rgb(200, 200, 200)'},
    name='Обучающая выборка'
))

fig.add_trace(go.Scatter(
    x=x_real,
    y=y_real,
    mode='lines',
    opacity=0.7,
    line={'dash': 'dash', 'color': 'orange'},
    name='Реальная зависимость'
))

fig.update_layout(width=600, height=500)

plot.plotly_chart(fig, use_container_width=True)


if is_start_learning:
    optimizer(
        X, y, np.ones(n_polinom + 1),
        loss=loss(regularizer(reg_coef), **loss_kw),
        learning_rate=lambda k: lmb * (s / (s + k)) ** p,
        stop_function=lambda _, k: k < n_iterations,
        callback=callback,
        **optimizer_kw
    )
