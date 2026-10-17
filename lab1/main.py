
import argparse

from data import load_data
from numpy_network import (
    initialize_parameters,
    forward,
    cross_entropy_loss,
)
from checks import run_all_checks


def main():
    parser = argparse.ArgumentParser(
        description="Лабораторна робота №1: зворотне поширення похибки"
    )
    parser.add_argument(
        "--bug",
        action="store_true",
        help="Запустити реалізацію з навмисною помилкою у градієнті",
    )
    args = parser.parse_args()

    print("Лабораторна робота №1")
    print("Ручне зворотне поширення похибки (Backpropagation)")

    if args.bug:
        print("\nЗ навмисною помилкою")
        print(
            "У градієнті за логітами пропущено ділення "
            "на кількість об'єктів."
        )
    else:
        print("\nПравильна реалізація(без навмисної помилки)")

    print("\n Підготовка даних ")

    X_train, y_train, X_test, y_test = load_data()

    print(f"Розмір навчальної вибірки: {X_train.shape}")
    print(f"Розмір тестової вибірки: {X_test.shape}")

    W1, b1, W2, b2 = initialize_parameters()

    _, _, logits = forward(X_train, W1, b1, W2, b2)
    loss = cross_entropy_loss(logits, y_train)

    print("\n Початкові параметри мережі")
    print("Архітектура: 4 -> 8 -> 3")
    print(f"Розмір W1: {W1.shape}")
    print(f"Розмір b1: {b1.shape}")
    print(f"Розмір W2: {W2.shape}")
    print(f"Розмір b2: {b2.shape}")
    print(f"Початкова функція втрат: {loss:.16f}")

    print("\n Перевірки ")
    run_all_checks(
        X_train,
        y_train,
        W1,
        b1,
        W2,
        b2,
        buggy=args.bug,
    )


if __name__ == "__main__":
    main()
