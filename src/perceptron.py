import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Perceptron


@dataclass
class Dataset:
    name: str
    feature_names: list[str]
    samples: np.ndarray
    labels: np.ndarray


@dataclass
class EpochLog:
    epoch: int
    errors: int
    accuracy: float
    coef: list[float]
    intercept: list[float]


@dataclass
class TrainingResult:
    dataset_name: str
    feature_names: list[str]
    eta0: float
    max_epochs: int
    random_state: int
    fit_intercept: bool
    shuffle: bool
    classes: list[int]
    epochs_run: int
    converged: bool
    samples: list[list[float]]
    labels: list[int]
    predictions: list[int]
    final_coef: list[float]
    final_intercept: list[float]
    epoch_logs: list[EpochLog]


def load_dataset(csv_path: Path) -> Dataset:
    rows: list[list[float]] = []
    labels: list[int] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "D" not in reader.fieldnames:
            raise ValueError(f"Arquivo {csv_path} precisa conter a coluna 'D'.")

        feature_names = [name for name in reader.fieldnames if name != "D"]

        for row in reader:
            rows.append([float(row[name]) for name in feature_names])
            label = int(float(row["D"]))
            if label not in (-1, 1):
                raise ValueError(
                    f"Classe invalida em {csv_path}. Use apenas valores -1 ou 1."
                )
            labels.append(label)

    if not rows:
        raise ValueError(f"Arquivo {csv_path} nao possui amostras.")

    return Dataset(
        name=csv_path.stem,
        feature_names=feature_names,
        samples=np.asarray(rows, dtype=float),
        labels=np.asarray(labels, dtype=int),
    )


def _should_fit_intercept(feature_names: list[str]) -> bool:
    return "Bias" not in {name.strip() for name in feature_names} # Força a reta plano


def train_perceptron(
    dataset: Dataset,
    eta0: float = 0.1,
    max_epochs: int = 100,
    random_state: int = 7,
    shuffle: bool = False,
) -> TrainingResult:
    fit_intercept = _should_fit_intercept(dataset.feature_names)
    model = Perceptron(
        penalty=None,
        alpha=0.0,
        eta0=eta0,
        max_iter=1,
        tol=None,
        shuffle=False,
        random_state=random_state,
        fit_intercept=fit_intercept, # Isso força a reta/plano a passar pela origem (0,0)
        warm_start=True,
    )
    classes = np.asarray(sorted(np.unique(dataset.labels)), dtype=int)
    rng = np.random.default_rng(random_state)
    epoch_logs: list[EpochLog] = []

    for epoch in range(1, max_epochs + 1):
        train_x = dataset.samples
        train_y = dataset.labels

        if shuffle:
            indices = rng.permutation(len(dataset.samples))
            train_x = dataset.samples[indices]
            train_y = dataset.labels[indices]

        if epoch == 1:
            model.partial_fit(train_x, train_y, classes=classes)
        else:
            model.partial_fit(train_x, train_y)

        predictions = model.predict(dataset.samples)
        errors = int(np.sum(predictions != dataset.labels))
        accuracy = float(np.mean(predictions == dataset.labels))
        coef = model.coef_.ravel().tolist()
        intercept = model.intercept_.ravel().tolist()
        epoch_logs.append(
            EpochLog(
                epoch=epoch,
                errors=errors,
                accuracy=accuracy,
                coef=coef,
                intercept=intercept,
            )
        )

        if errors == 0:
            return TrainingResult(
                dataset_name=dataset.name,
                feature_names=dataset.feature_names,
                eta0=eta0,
                max_epochs=max_epochs,
                random_state=random_state,
                fit_intercept=fit_intercept,
                shuffle=shuffle,
                classes=classes.tolist(),
                epochs_run=epoch,
                converged=True,
                samples=dataset.samples.tolist(),
                labels=dataset.labels.tolist(),
                predictions=predictions.astype(int).tolist(),
                final_coef=coef,
                final_intercept=intercept,
                epoch_logs=epoch_logs,
            )

    predictions = model.predict(dataset.samples)
    return TrainingResult(
        dataset_name=dataset.name,
        feature_names=dataset.feature_names,
        eta0=eta0,
        max_epochs=max_epochs,
        random_state=random_state,
        fit_intercept=fit_intercept,
        shuffle=shuffle,
        classes=classes.tolist(),
        epochs_run=max_epochs,
        converged=False,
        samples=dataset.samples.tolist(),
        labels=dataset.labels.tolist(),
        predictions=predictions.astype(int).tolist(),
        final_coef=model.coef_.ravel().tolist(),
        final_intercept=model.intercept_.ravel().tolist(),
        epoch_logs=epoch_logs,
    )
