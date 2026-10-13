"""Разбор MNIST до обучения.

Скачивает картинки, печатает, из чего они состоят, и сохраняет сетку примеров.
Сеть здесь не создаётся и не учится. Скрипт только показывает две части архива:

* 60 000 обучающих картинок — на них позже будут меняться веса;
* 10 000 других картинок — их в обучение не кладут, на них потом проверяют,
  узнаёт ли сеть новую цифру, а не запомненный пример.
"""

import sys

import matplotlib

# Agg рисует в файл и не открывает окно. Так скрипт работает без экрана.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from model import DATA_DIR, OUTPUT_DIR

# Сколько примеров одной цифры класть в сетку.
SAMPLES_PER_DIGIT = 5
# Символы от тёмного пикселя к светлому для текстового наброска.
ASCII_RAMP = " .:-=+*#%@"


def ascii_digit(image):
    """Набросок одной картинки (1, 28, 28) моноширинными символами."""
    pixels = image.squeeze(0)
    lines = []
    for row in pixels:
        characters = []
        for value in row:
            # Яркость 0…1 переводим в индекс символа.
            index = min(len(ASCII_RAMP) - 1, int(float(value) * len(ASCII_RAMP)))
            characters.append(ASCII_RAMP[index])
        lines.append("".join(characters))
    return "\n".join(lines)


def pixel_rows(image, how_many):
    """Несколько строк пикселей там, где штрих уже начался.

    Сверху у цифры пустое поле, и первые строки файла были бы одними нулями.
    Берём первую строку с ненулевой яркостью и ещё how_many-1 следом.
    """
    pixels = image.squeeze(0)
    start = 0
    for index, row in enumerate(pixels):
        if float(row.max()) > 0:
            start = index
            break
    chosen = pixels[start:start + how_many]
    lines = [f"  строки {start}–{start + chosen.shape[0] - 1} (нумерация с нуля):"]
    for row in chosen:
        lines.append(" ".join(f"{float(value):.2f}" for value in row))
    return "\n".join(lines)


def class_counts(targets):
    """Сколько картинок каждой цифры 0–9."""
    counts = torch.bincount(targets, minlength=10)
    return [int(count) for count in counts]


def format_counts(counts):
    return "\n".join(f"  {digit}: {count}" for digit, count in enumerate(counts))


def first_indexes_per_digit(targets, per_digit):
    """Первые per_digit индексов каждой метки, в порядке файла, не «как попало»."""
    found = {digit: [] for digit in range(10)}
    for index, label in enumerate(targets.tolist()):
        bucket = found[int(label)]
        if len(bucket) < per_digit:
            bucket.append(index)
        if all(len(bucket) == per_digit for bucket in found.values()):
            break
    return found


def save_sample_grid(train_set, indexes, destination):
    """Сетка 10 строк × SAMPLES_PER_DIGIT. Строка — одна цифра."""
    figure, axes = plt.subplots(
        10,
        SAMPLES_PER_DIGIT,
        figsize=(SAMPLES_PER_DIGIT * 1.3, 14),
    )
    for digit in range(10):
        for column, index in enumerate(indexes[digit]):
            ax = axes[digit, column]
            # squeeze убирает канал: (1, 28, 28) → (28, 28).
            # vmin/vmax фиксируют шкалу 0…1, чтобы картинки можно было сравнивать.
            ax.imshow(train_set[index][0].squeeze(0), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_ylabel(str(digit), rotation=0, labelpad=12, fontsize=12)
    figure.suptitle("MNIST, обучающая часть: первые примеры каждой цифры")
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)


def build_report(train_set, test_set):
    train_image, train_label = train_set[0]
    test_image, test_label = test_set[0]

    # У MNIST в памяти лежат целые 0…255. ToTensor делит их на 255.
    # Считаем минимум и максимум по всей обучающей части уже после этого деления,
    # то есть в тех числах, которые потом увидит сеть.
    brightness = train_set.data.float() / 255.0
    test_brightness = test_set.data.float() / 255.0

    # 28×28 вытягивается в 784 в том же порядке, в каком нейрон читает вход.
    flat = train_image.view(-1)

    lines = [
        "MNIST: рукописные цифры",
        "",
        "Архив разделён на две непересекающиеся части.",
        "Обучающая (train=True): на этих картинках позже обновляются веса.",
        "Проверочная (train=False): эти картинки в обучение не входят.",
        "На них сеть только экзаменуют, чтобы увидеть, узнаёт ли она новые цифры,",
        "а не те, которые уже запомнила.",
        "",
        f"Обучающих картинок: {len(train_set)}",
        f"Проверочных картинок: {len(test_set)}",
        "",
        "Одна картинка после ToTensor:",
        f"  форма: {tuple(train_image.shape)}  (канал, высота, ширина)",
        f"  dtype: {train_image.dtype}",
        (
            "  яркость обучающей части: от "
            f"{float(brightness.min()):.1f} до {float(brightness.max()):.1f}"
        ).replace(".", ","),
        (
            "  яркость проверочной части: от "
            f"{float(test_brightness.min()):.1f} до {float(test_brightness.max()):.1f}"
        ).replace(".", ","),
        "",
        "Сколько картинок каждой цифры в обучающей части:",
        format_counts(class_counts(train_set.targets)),
        "",
        "Сколько картинок каждой цифры в проверочной части:",
        format_counts(class_counts(test_set.targets)),
        "",
        "Первый обучающий пример и первый проверочный — разные картинки.",
        f"  обучающий: цифра {int(train_label)}",
        f"  проверочный: цифра {int(test_label)}",
        "",
        "Четыре строки пикселей первого обучающего примера, с первого штриха (яркость 0…1):",
        pixel_rows(train_image, 4),
        "",
        "Четыре строки первого проверочного примера, с первого штриха:",
        pixel_rows(test_image, 4),
        "",
        "Набросок первого обучающего примера. Точка темнее, @ светлее:",
        ascii_digit(train_image),
        "",
        "Как это ложится на сеть из ролика.",
        "Квадрат 28×28 вытягивается в один вектор слева направо, сверху вниз.",
        f"  длина вектора: {flat.shape[0]}",
        "Каждое из этих чисел попадёт на свой входной нейрон.",
        "У первого скрытого слоя 16 нейронов, у каждого по 784 веса —",
        "по весу на каждый пиксель. Эти 784 веса можно снова выложить",
        "квадратом 28×28 и увидеть, на какой узор смотрит нейрон.",
        "",
    ]
    return "\n".join(lines)


def main():
    # Консоль Windows по умолчанию не utf-8, а в отчёте есть «×».
    sys.stdout.reconfigure(encoding="utf-8")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    report = build_report(train_set, test_set)
    report_path = OUTPUT_DIR / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Отчёт записан в {report_path}")

    indexes = first_indexes_per_digit(train_set.targets, SAMPLES_PER_DIGIT)
    grid_path = OUTPUT_DIR / "samples.png"
    save_sample_grid(train_set, indexes, grid_path)
    print(f"Сетка примеров записана в {grid_path}")


if __name__ == "__main__":
    main()
