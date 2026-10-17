
import numpy as np


def initialize_parameters():

    rng = np.random.default_rng(0)

    # 1 шар - 4 вхідні ознаки -> 8 нейронів
    W1 = rng.normal(
        loc=0.0,
        scale=np.sqrt(2.0 / 4),
        size=(4, 8),
    ).astype(np.float64)
    b1 = np.zeros(8, dtype=np.float64)

    # 2 шар - 8 нейронів -> 3 вихідні класи
    W2 = rng.normal(
        loc=0.0,
        scale=np.sqrt(2.0 / (8 + 3)),
        size=(8, 3),
    ).astype(np.float64)
    b2 = np.zeros(3, dtype=np.float64)

    return W1, b1, W2, b2

# forward виконує прямий прохід нейронної мережі
def forward(X, W1, b1, W2, b2):

    # 1 лінійний шар
    Z1 = X @ W1 + b1

    # Функція активації ReLU
    A1 = np.maximum(0.0, Z1)

    # 2 лінійний шар
    logits = A1 @ W2 + b2

    return Z1, A1, logits


def cross_entropy_loss(logits, y):

    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)

    log_sum_exp = np.log(
        np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    )

    log_probs = shifted_logits - log_sum_exp

    correct_class_log_probs = log_probs[np.arange(len(y)), y]

    # Середня функція втрат для всіх об'єктів
    return -np.mean(correct_class_log_probs)


def backward(
    X, y, Z1, A1, logits, W1, W2,
    omit_batch_division=False,
):

    n = X.shape[0]

    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    d_logits = probs.copy()
    d_logits[np.arange(n), y] -= 1.0

    
    # для перевірки з навмисною помилкою
    if not omit_batch_division:
        d_logits /= n

    dW2 = A1.T @ d_logits
    db2 = np.sum(d_logits, axis=0)

    dA1 = d_logits @ W2.T

    # Похідна ReLU = 1 там, де Z1 > 0, і 0 в інших випадках
    dZ1 = dA1 * (Z1 > 0)

    # Обчислення градієнти першого лінійного шару
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
    }


if __name__ == "__main__":
    from data import load_data

    X_train, y_train, X_test, y_test = load_data()

    W1, b1, W2, b2 = initialize_parameters()
    Z1, A1, logits = forward(X_train, W1, b1, W2, b2)
    loss = cross_entropy_loss(logits, y_train)

    grads = backward(X_train, y_train, Z1, A1, logits, W1, W2)

    print("Функція втрат:", loss)

    for name, grad in grads.items():
        print(
            f"Градієнт {name}: "
            f"форма={grad.shape}, "
            f"усі значення скінченні={np.isfinite(grad).all()}, "
            f"максимальне абсолютне значення="
            f"{np.max(np.abs(grad)):.8e}"
        )

    assert grads["W1"].shape == W1.shape
    assert grads["b1"].shape == b1.shape
    assert grads["W2"].shape == W2.shape
    assert grads["b2"].shape == b2.shape

    assert all(np.isfinite(grad).all() for grad in grads.values())
