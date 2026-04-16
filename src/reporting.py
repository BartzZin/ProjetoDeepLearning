import csv
import json
from pathlib import Path

from src.perceptron import TrainingResult


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_predictions(result: TrainingResult, output_dir: Path) -> Path:
    ensure_directory(output_dir)
    csv_path = output_dir / f"{result.dataset_name}_predicoes.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["amostra", "atributos", "classe_real", "classe_prevista", "acerto"]
        )

        for index, (sample, label, prediction) in enumerate(
            zip(result.samples, result.labels, result.predictions), start=1
        ):
            attributes = " | ".join(
                f"{name}={value}"
                for name, value in zip(result.feature_names, sample)
            )
            writer.writerow(
                [
                    index,
                    attributes,
                    label,
                    prediction,
                    "sim" if prediction == label else "nao",
                ]
            )

    return csv_path


def save_history(result: TrainingResult, output_dir: Path) -> Path:
    ensure_directory(output_dir)
    csv_path = output_dir / f"{result.dataset_name}_historico_epocas.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoca", "erros", "acuracia", "coeficientes", "intercepto"])

        for log in result.epoch_logs:
            writer.writerow(
                [
                    log.epoch,
                    log.errors,
                    log.accuracy,
                    json.dumps(log.coef),
                    json.dumps(log.intercept),
                ]
            )

    return csv_path


def save_summary(result: TrainingResult, output_dir: Path) -> Path:
    ensure_directory(output_dir)
    json_path = output_dir / f"{result.dataset_name}_resumo.json"
    payload = {
        "dataset": result.dataset_name,
        "feature_names": result.feature_names,
        "eta0": result.eta0,
        "maximo_epocas": result.max_epochs,
        "random_state": result.random_state,
        "fit_intercept": result.fit_intercept,
        "shuffle": result.shuffle,
        "classes": result.classes,
        "coeficientes_finais": result.final_coef,
        "intercepto_final": result.final_intercept,
        "epocas_executadas": result.epochs_run,
        "convergiu": result.converged,
        "total_amostras": len(result.labels),
        "acertos": sum(
            1 for label, prediction in zip(result.labels, result.predictions)
            if label == prediction
        ),
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)

    return json_path
