import torch
import numpy as np
from sklearn.model_selection import KFold


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion, apply_max_norm, max_norm_val, device):

    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch_x, batch_y in iterator:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()  # gradient 초기화
        predictions = model(batch_x).squeeze(1)
        loss = criterion(predictions, batch_y)
        acc = binary_accuracy(predictions, batch_y)
        loss.backward()  # backpropagation
        optimizer.step()  # step check

        if apply_max_norm == True:
            param = model.fc.weight.norm()
            eps = 1e-5
            norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, max_norm_val)
            param.data = param * (desired / (eps + norm))

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device="cpu"):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()  # 모델 로드
    with torch.no_grad():  # test시에만 작동
        for batch_x, batch_y in iterator:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            predictions = model(batch_x).squeeze(1)
            loss = criterion(predictions, batch_y)

            acc = binary_accuracy(predictions, batch_y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
