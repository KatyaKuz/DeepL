
import numpy as np
import torch
from torch import nn

# Функція втрат і градієнти еталонної мережі PyTorch
def evaluate_with_torch(X, y, W1, b1, W2, b2):

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    ).double()

    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(W1.T.copy()))
        model[0].bias.copy_(torch.from_numpy(b1.copy()))
        model[2].weight.copy_(torch.from_numpy(W2.T.copy()))
        model[2].bias.copy_(torch.from_numpy(b2.copy()))

    X_torch = torch.from_numpy(X.copy())
    y_torch = torch.from_numpy(y.copy()).long()

    # Прямий прохід і функція втрат
    logits = model(X_torch)
    loss = nn.functional.cross_entropy(logits, y_torch)

    # Диференціювання
    loss.backward()

    gradients = {
        "W1": model[0].weight.grad.detach().numpy().T.copy(),
        "b1": model[0].bias.grad.detach().numpy().copy(),
        "W2": model[2].weight.grad.detach().numpy().T.copy(),
        "b2": model[2].bias.grad.detach().numpy().copy(),
    }

    return loss.item(), gradients
