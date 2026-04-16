import csv
from pathlib import Path

from sklearn.datasets import load_iris


def ensure_iris_datasets(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    labels = [1 if target == 0 else -1 for target in iris.target]

    iris_4d_path = output_dir / "iris_binario_4d.csv"
    iris_2d_path = output_dir / "iris_binario_2d.csv"

    with iris_4d_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Bias", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "D"])
        for features, label in zip(iris.data, labels):
            writer.writerow([1, *features.tolist(), label]) # Bias Fixo em 1

    with iris_2d_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Bias", "PetalLength", "PetalWidth", "D"])
        for features, label in zip(iris.data, labels):
            writer.writerow([1, features[2], features[3], label]) # Bias fixo em 1

    return [iris_4d_path, iris_2d_path]
