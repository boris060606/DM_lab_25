import os
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

from ucimlrepo import fetch_ucirepo


#  Параметры 

SEED = 42
BATCH_SIZE = 128
PRETRAIN_EPOCHS = 30
TRAIN_EPOCHS = 100
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs("results/digits", exist_ok=True)
os.makedirs("results/cardiotocography", exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Устройство:", DEVICE)


#  Загрузка данных 

def load_digits_data():
    data = load_digits()

    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    return X, y, [str(i) for i in range(10)]


def load_cardiotocography_data():
    # UCI Cardiotocography, dataset ID 193
    dataset = fetch_ucirepo(id=193)

    X = dataset.data.features.copy()
    y_data = dataset.data.targets.copy()

    # Для работы используем целевую переменную NSP.
    target_column = next(
        (c for c in y_data.columns
         if c.strip().lower() == "nsp"),
        None
    )

    if target_column is None:
        raise ValueError(
            "В целевых переменных не найдена NSP. "
            "Проверьте названия столбцов датасета."
        )

    # Преобразование признаков в числовой формат.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Заполняем пропуски медианами признаков.
    X = X.fillna(X.median()).fillna(0)

    # Кодируем целевые метки.
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_data[target_column])

    return (
        X.to_numpy(dtype=np.float32),
        y.astype(np.int64),
        [str(c) for c in encoder.classes_]
    )


#  Подготовка данных 

def prepare_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=SEED,
        stratify=y
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp
    )

    # Масштабирование обучается только на train.
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    def make_loader(features, labels, shuffle=False):
        dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )

        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle
        )

    train_loader = make_loader(X_train, y_train, True)
    valid_loader = make_loader(X_valid, y_valid)
    test_loader = make_loader(X_test, y_test)

    return train_loader, valid_loader, test_loader, X_train


#  Автоэнкодер 

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


def pretrain_layer(data, input_size, hidden_size):
    autoencoder = Autoencoder(
        input_size, hidden_size
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        autoencoder.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(data, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    autoencoder.train()

    for epoch in range(PRETRAIN_EPOCHS):
        total_loss = 0

        for (batch,) in loader:
            batch = batch.to(DEVICE)

            reconstructed = autoencoder(batch)
            loss = criterion(reconstructed, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)

        if (epoch + 1) % 10 == 0:
            print(
                f"  Предобучение: эпоха {epoch + 1}, "
                f"MSE = {total_loss / len(data):.5f}"
            )

    autoencoder.eval()

    with torch.no_grad():
        encoded = autoencoder.encoder(
            torch.tensor(data, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()

    return autoencoder.encoder[0], encoded


#  Классификатор 

class Classifier(nn.Module):
    def __init__(self, input_size, class_count):
        super().__init__()

        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, class_count)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output(x)


def initialize_with_pretraining(model, X_train):
    print("Предобучение первого скрытого слоя...")

    layer1, representation1 = pretrain_layer(
        X_train, X_train.shape[1], 128
    )

    model.layer1.load_state_dict(layer1.state_dict())

    print("Предобучение второго скрытого слоя...")

    layer2, _ = pretrain_layer(
        representation1, 128, 64
    )

    model.layer2.load_state_dict(layer2.state_dict())


#  Обучение классификатора 

def train_model(model, train_loader, valid_loader, name, folder):
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_accuracy": [],
        "valid_accuracy": []
    }

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(TRAIN_EPOCHS):
        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

            predictions = logits.argmax(dim=1)
            train_correct += (
                predictions == y_batch
            ).sum().item()

            train_total += y_batch.size(0)

        model.eval()

        valid_loss = 0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                valid_loss += loss.item() * X_batch.size(0)

                predictions = logits.argmax(dim=1)
                valid_correct += (
                    predictions == y_batch
                ).sum().item()

                valid_total += y_batch.size(0)

        train_loss /= train_total
        valid_loss /= valid_total

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        history["train_accuracy"].append(
            train_correct / train_total
        )
        history["valid_accuracy"].append(
            valid_correct / valid_total
        )

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            print(
                f"{name}: эпоха {epoch + 1}, "
                f"train loss={train_loss:.4f}, "
                f"valid loss={valid_loss:.4f}, "
                f"valid accuracy={valid_correct / valid_total:.4f}"
            )

    model.load_state_dict(best_state)

    torch.save(
        model.state_dict(),
        os.path.join(folder, f"{name}_model.pth")
    )

    return model, history


#  Оценка качества 

def evaluate_model(model, test_loader, class_names, name, folder):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)

            predictions = model(X_batch).argmax(dim=1)

            y_true.extend(y_batch.numpy())
            y_pred.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)

    f1_macro = f1_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )

    display.plot(
        xticks_rotation=45,
        cmap="Blues"
    )

    plt.title(f"Матрица ошибок: {name}")
    plt.tight_layout()

    plt.savefig(
        os.path.join(folder, f"{name}_confusion_matrix.png")
    )

    plt.close()

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_f1": f1_macro
    }


#  Графики обучения 

def save_history(history, name, folder):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["valid_loss"], label="Validation")
    plt.xlabel("Эпоха")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Функция потерь: {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{name}_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train"
    )

    plt.plot(
        epochs,
        history["valid_accuracy"],
        label="Validation"
    )

    plt.xlabel("Эпоха")
    plt.ylabel("Accuracy")
    plt.title(f"Точность: {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{name}_accuracy.png"))
    plt.close()


#  Эксперимент 

def run_experiment(dataset_name, X, y, class_names):
    folder = os.path.join("results", dataset_name)
    os.makedirs(folder, exist_ok=True)

    print("Датасет:", dataset_name)
    print("Размер:", X.shape)
    print("Количество классов:", len(class_names))

    train_loader, valid_loader, test_loader, X_train = (
        prepare_data(X, y)
    )

    results = []

    # 1. Обучение с нуля.
    print("\nОбучение без предобучения...")

    model_plain = Classifier(
        X.shape[1], len(class_names)
    )

    model_plain, history_plain = train_model(
        model_plain,
        train_loader,
        valid_loader,
        "without_pretraining",
        folder
    )

    save_history(
        history_plain, "without_pretraining", folder
    )
    results.append(
        evaluate_model(
            model_plain,
            test_loader,
            class_names,
            "without_pretraining",
            folder
        )
    )

    # 2. Предобучение автоэнкодерами.
    print("\nОбучение с предобучением...")

    model_pretrained = Classifier(
        X.shape[1], len(class_names)
    )

    initialize_with_pretraining(
        model_pretrained, X_train
    )

    model_pretrained, history_pretrained = train_model(
        model_pretrained,
        train_loader,
        valid_loader,
        "with_pretraining",
        folder
    )

    save_history(
        history_pretrained, "with_pretraining", folder
    )

    results.append(
        evaluate_model(
            model_pretrained,
            test_loader,
            class_names,
            "with_pretraining",
            folder
        )
    )

    return results


#  Главная функция 

def main():
    all_results = []

    # Датасет из ЛР 2.
    X_digits, y_digits, digits_classes = load_digits_data()

    all_results.extend(
        run_experiment(
            "digits",
            X_digits,
            y_digits,
            digits_classes
        )
    )

    # Датасет из ЛР 3.
    X_ctg, y_ctg, ctg_classes = load_cardiotocography_data()

    all_results.extend(
        run_experiment(
            "cardiotocography",
            X_ctg,
            y_ctg,
            ctg_classes
        )
    )

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        "results/comparison.csv",
        index=False
    )

    print("\nИтоговое сравнение:")
    print(results_df.to_string(index=False))

    print("\nРезультаты сохранены в папке results.")


if __name__ == "__main__":
    main()
