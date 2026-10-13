"""Обучение сети на 60 000 картинках MNIST.

Этот файл подготовлен заранее и в шаге разбора данных не запускается.
Когда его запустят, веса меняются только на обучающей части.
Отложенные 10 000 картинок в loss.backward() не попадают: по ним считают
долю верных ответов, но градиент по ним не идёт.

Доля на этих 10 000 — журнал при уже выбранных настройках.
По ней не подбирают скорость обучения и число эпох: иначе экзамен
перестанет быть честным. Для подбора нужна была бы третья выборка.
"""

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import (
    BATCH_SIZE,
    CHECKPOINT_EVERY_BATCHES,
    CHECKPOINT_PATH,
    DATA_DIR,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT_DIR,
    SEED,
    DigitNet,
)


def save_checkpoint(model, epoch, batch, test_accuracy):
    """Атомарно записать снимок, который читает страница весов.

    Сначала пишем временный файл, потом подменяем им постоянный.
    Читатель либо видит старый целый снимок, либо новый целый,
    но не обрывки посередине записи.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "batch": batch,
        "test_accuracy": test_accuracy,
    }
    temporary = CHECKPOINT_PATH.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, CHECKPOINT_PATH)


def evaluate(model, loader):
    """Доля верных ответов. Веса здесь не меняются.

    model.eval() нужен, если позже появятся слои вроде Dropout:
    на экзамене они должны вести себя иначе, чем при обучении.
    torch.no_grad() не строит граф ошибки, поэтому backward()
    от этих картинок вызвать нельзя даже случайно.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            predicted = model(images).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.shape[0]
    return correct / total


def main():
    # Seed до создания сети, чтобы стартовые веса совпали со страницей,
    # которую открыли до обучения.
    torch.manual_seed(SEED)
    model = DigitNet()

    # ToTensor переводит пиксель 0…255 в яркость 0…1 и форму (1, 28, 28).
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    # Обучающий загрузчик перемешивает: иначе сеть много шагов подряд
    # видела бы цифры в одном и том же порядке файла.
    # Проверочный не перемешиваем, на ответ это не влияет.
    # num_workers оставляем 0: на Windows лишние процессы загрузчика
    # часто падают при запуске из скрипта.
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    # Метки — целые 0–9, не вектор из десяти чисел, как в ролике.
    # Это то же соответствие «картинка → цифра», только короче.
    # CrossEntropyLoss сам переводит метку в сравнение с десятью оценками.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # None, пока первая эпоха не дошла до экзамена.
    test_accuracy = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        last_batch = 0
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            last_batch = batch_index
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            # Градиент только по этому обучающему батчу.
            loss.backward()
            optimizer.step()

            if batch_index % CHECKPOINT_EVERY_BATCHES == 0:
                save_checkpoint(model, epoch, batch_index, test_accuracy)

        # Экзамен после прохода по всем 60 000. В обновление весов не входит.
        test_accuracy = evaluate(model, test_loader)
        print(
            f"Эпоха {epoch}: верных ответов на проверочных 10 000 — "
            f"{test_accuracy:.1%}"
        )
        save_checkpoint(model, epoch, last_batch, test_accuracy)


if __name__ == "__main__":
    main()
