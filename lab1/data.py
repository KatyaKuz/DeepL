
import numpy as np
from sklearn.datasets import load_iris


def load_data():
    """Завантажує набір Iris, розділяє дані та стандартизує ознаки."""

    # Вихідні дані.
    iris = load_iris()

    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int64)

    # Генератор із фіксованим початковим значенням
    rng = np.random.default_rng(0)

    train_indices = []
    test_indices = []

    for class_id in (0, 1, 2):
        class_indices = np.flatnonzero(y == class_id)
        shuffled_indices = rng.permutation(class_indices)

        train_indices.extend(shuffled_indices[:35])
        test_indices.extend(shuffled_indices[35:])

    train_indices = np.array(train_indices, dtype=np.int64)
    test_indices = np.array(test_indices, dtype=np.int64)

    # Навчальна та тестова вибірки
    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Перевірка розмірів та коректності даних
    assert X_train.shape == (105, 4)
    assert y_train.shape == (105,)
    assert X_test.shape == (45, 4)
    assert y_test.shape == (45,)

    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()

    for class_id in (0, 1, 2):
        assert np.count_nonzero(y_train == class_id) == 35
        assert np.count_nonzero(y_test == class_id) == 15

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()

    print("Розмір навчальної вибірки:", X_train.shape)
    print("Розмір тестової вибірки:", X_test.shape)
    print("Кількість об'єктів кожного класу в навчальній вибірці:",
          np.bincount(y_train))
    print("Кількість об'єктів кожного класу в тестовій вибірці:",
          np.bincount(y_test))
    print("Середні значення ознак навчальної вибірки:",
          X_train.mean(axis=0))
    print("Стандартні відхилення ознак навчальної вибірки:",
          X_train.std(axis=0))
    print("Усі перевірки пройдено.")
