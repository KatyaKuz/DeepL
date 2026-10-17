
import numpy as np

from numpy_network import (
    forward,
    cross_entropy_loss,
    backward,
)
from torch_ref import evaluate_with_torch


TOL_TORCH = 1e-12
TOL_NUMERICAL = 1e-7
EPSILON = 1e-6


# Ручний метод зворотного поширення
def get_gradients(X, y, W1, b1, W2, b2, omit_batch_division=False):
    Z1, A1, logits = forward(X, W1, b1, W2, b2)

    return backward(
        X,
        y,
        Z1,
        A1,
        logits,
        W1,
        W2,
        omit_batch_division=omit_batch_division,
    )


# Функція втрат для заданих параметрів
def calculate_loss(X, y, W1, b1, W2, b2):
    _, _, logits = forward(X, W1, b1, W2, b2)
    return cross_entropy_loss(logits, y)


# Чисельна похідна методом центральних різниць
def numerical_gradient(X, y, parameters, name, index):
    parameter = parameters[name]
    original_value = parameter[index]

    try:
        parameter[index] = original_value + EPSILON
        loss_plus = calculate_loss(
            X, y,
            parameters["W1"],
            parameters["b1"],
            parameters["W2"],
            parameters["b2"],
        )

        parameter[index] = original_value - EPSILON
        loss_minus = calculate_loss(
            X, y,
            parameters["W1"],
            parameters["b1"],
            parameters["W2"],
            parameters["b2"],
        )
    finally:
        parameter[index] = original_value

    return (loss_plus - loss_minus) / (2.0 * EPSILON)


# Порівняння функцій втрат і градієнтів NumPy та PyTorch
def check_torch(X, y, parameters):
    print("\nПеревірка NumPy та PyTorch")

    W1, b1, W2, b2 = (
        parameters["W1"],
        parameters["b1"],
        parameters["W2"],
        parameters["b2"],
    )

    loss_numpy = calculate_loss(X, y, W1, b1, W2, b2)
    loss_torch, gradients_torch = evaluate_with_torch(
        X, y, W1, b1, W2, b2
    )

    gradients_numpy = get_gradients(X, y, W1, b1, W2, b2)

    loss_diff = abs(loss_numpy - loss_torch)

    print(f"Функція втрат NumPy:   {loss_numpy:.16f}")
    print(f"Функція втрат PyTorch: {loss_torch:.16f}")
    print(f"Абсолютна різниця:     {loss_diff:.3e}")

    assert np.isfinite(loss_numpy)
    assert np.isfinite(loss_torch)
    assert loss_diff <= TOL_TORCH, (
        "Функції втрат NumPy та PyTorch не збігаються."
    )

    for name in ("W1", "b1", "W2", "b2"):
        grad_numpy = gradients_numpy[name]
        grad_torch = gradients_torch[name]

        finite = (
            np.all(np.isfinite(grad_numpy))
            and np.all(np.isfinite(grad_torch))
        )

        assert finite, (
            f"Градієнти {name} містять нескінченні значення або NaN."
        )

        diff = np.max(np.abs(grad_numpy - grad_torch))

        print(
            f"{name}: максимальна абсолютна різниця = "
            f"{diff:.3e}; скінченні значення = {finite}"
        )

        assert diff <= TOL_TORCH, (
            f"Градієнти {name} NumPy та PyTorch не збігаються."
        )


# Перевірка чотирьох градієнтів методом центральних різниць
def check_finite_differences(X, y, parameters):
    print("\nПеревірка методом скінченних різниць")

    gradients = get_gradients(
        X,
        y,
        parameters["W1"],
        parameters["b1"],
        parameters["W2"],
        parameters["b2"],
    )

    elements = (
        ("W1", (0, 0)),
        ("b1", (0,)),
        ("W2", (0, 0)),
        ("b2", (0,)),
    )

    for name, index in elements:
        manual = gradients[name][index]
        numeric = numerical_gradient(X, y, parameters, name, index)
        diff = abs(manual - numeric)

        print(
            f"{name}{index}: ручний градієнт = {manual:+.12e}; "
            f"чисельна похідна = {numeric:+.12e}; "
            f"різниця = {diff:.3e}"
        )

        assert np.isfinite(manual) and np.isfinite(numeric)
        assert diff <= TOL_NUMERICAL, (
            f"Перевірку скінченними різницями не пройдено: "
            f"{name}{index}."
        )


# Перевірка помилкової реалізації та множника 105
def check_buggy_backward(X, y, parameters):
    print("\nПеревірка помилкової реалізації")

    W1, b1, W2, b2 = (
        parameters["W1"],
        parameters["b1"],
        parameters["W2"],
        parameters["b2"],
    )

    # Функція втрат не залежить від способу обчислення градієнтів.
    loss_numpy = calculate_loss(X, y, W1, b1, W2, b2)
    loss_torch, gradients_torch = evaluate_with_torch(
        X, y, W1, b1, W2, b2
    )

    print(f"Функція втрат NumPy:   {loss_numpy:.16f}")
    print(f"Функція втрат PyTorch: {loss_torch:.16f}")
    print(f"Різниця функцій втрат: {abs(loss_numpy - loss_torch):.3e}")

    assert np.isfinite(loss_numpy)
    assert np.isfinite(loss_torch)
    assert abs(loss_numpy - loss_torch) <= TOL_TORCH, (
        "Функції втрат NumPy та PyTorch не збігаються."
    )

    # Перевірка скінченності градієнтів PyTorch.
    for name in ("W1", "b1", "W2", "b2"):
        assert np.all(np.isfinite(gradients_torch[name])), (
            f"Градієнти PyTorch {name} містять нескінченні значення або NaN."
        )

    # Обчислення правильних і помилкових градієнтів.
    correct_gradients = get_gradients(X, y, W1, b1, W2, b2)

    buggy_gradients = get_gradients(
        X, y, W1, b1, W2, b2,
        omit_batch_division=True,
    )

    print("\nПеревірка множника 105")

    for name in ("W1", "b1", "W2", "b2"):
        correct = correct_gradients[name]
        buggy = buggy_gradients[name]

        # Без ділення на N градієнти збільшуються в N = 105 разів.
        expected_buggy = 105.0 * correct

        diff_scale = np.max(np.abs(buggy - expected_buggy))

        assert np.all(np.isfinite(correct))
        assert np.all(np.isfinite(buggy))
        assert np.allclose(
            buggy,
            expected_buggy,
            rtol=1e-12,
            atol=1e-12,
        ), (
            f"Градієнт {name} не дорівнює правильному "
            "градієнту, помноженому на 105."
        )

        # Нульові елементи мають залишитися нульовими.
        zero_mask = (correct == 0.0)

        assert np.all(buggy[zero_mask] == 0.0), (
            f"Нульові елементи градієнта {name} змінилися."
        )

        # Помилковий градієнт не повинен збігатися з PyTorch.
        diff_torch = np.max(
            np.abs(buggy - gradients_torch[name])
        )

        print(
            f"{name}: максимальна різниця з 105 × "
            f"правильний градієнт = {diff_scale:.3e}; "
            f"різниця з PyTorch = {diff_torch:.3e}"
        )

        assert diff_torch > TOL_NUMERICAL, (
            f"Помилку не виявлено для градієнта {name}."
        )

    # Порівняння помилкових градієнтів із чисельними похідними.
    print("\nПеревірка чисельними похідними")

    elements = (
        ("W1", (0, 0)),
        ("b1", (0,)),
        ("W2", (0, 0)),
        ("b2", (0,)),
    )

    for name, index in elements:
        numeric = numerical_gradient(
            X, y, parameters, name, index
        )
        buggy = buggy_gradients[name][index]
        diff = abs(numeric - buggy)

        print(
            f"{name}{index}: чисельна похідна = {numeric:+.8e}; "
            f"помилковий градієнт = {buggy:+.8e}; "
            f"різниця = {diff:.3e}"
        )

        assert np.isfinite(numeric) and np.isfinite(buggy)
        assert diff > TOL_NUMERICAL, (
            f"Чисельна перевірка не виявила помилку для {name}{index}."
        )

    print("\nПомилку успішно виявлено.")
    print("Функція втрат не змінилася.")
    print("Усі чотири градієнти перевірено: множник 105 підтверджено.")
    print("Порівняння з PyTorch і чисельними похідними виявили помилку.")


# Запуск перевірок відповідно до обраного режиму
def run_all_checks(X, y, W1, b1, W2, b2, buggy=False):
    parameters = {
        "W1": W1.copy(),
        "b1": b1.copy(),
        "W2": W2.copy(),
        "b2": b2.copy(),
    }

    if buggy:
        # Перевірка, чи виявляють тести навмисну помилку.
        check_buggy_backward(X, y, parameters)
    else:
        # Перевірка правильної реалізації.
        check_torch(X, y, parameters)
        check_finite_differences(X, y, parameters)
