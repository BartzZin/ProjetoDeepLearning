from pathlib import Path

from src.iris_datasets import ensure_iris_datasets
from src.perceptron import load_dataset, train_perceptron
from src.reporting import save_history, save_predictions, save_summary


BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "Bases"
OUTPUT_DIR = BASE_DIR / "Resultados"
ETA0 = 0.1
MAX_EPOCHS = 100
RANDOM_STATE = 42 #Professor de machinelearning falou que é um numero magico hehe
SHUFFLE = False


def process_dataset(
    csv_path: Path,
    eta0: float = ETA0,
    max_epochs: int = MAX_EPOCHS,
    random_state: int = RANDOM_STATE,
    shuffle: bool = SHUFFLE,
) -> str:
    dataset = load_dataset(csv_path)
    result = train_perceptron(
        dataset=dataset,
        eta0=eta0,
        max_epochs=max_epochs,
        random_state=random_state,
        shuffle=shuffle,
    )

    save_predictions(result, OUTPUT_DIR)
    save_history(result, OUTPUT_DIR)
    save_summary(result, OUTPUT_DIR)

    status = "convergiu" if result.converged else "nao convergiu"
    return (
        f"{csv_path.name}: {status} em {result.epochs_run} epoca(s). "
        f"coef_ = {result.final_coef}, intercept_ = {result.final_intercept}"
    )


def run_all_datasets(
    eta0: float = ETA0,
    max_epochs: int = MAX_EPOCHS,
    random_state: int = RANDOM_STATE,
    shuffle: bool = SHUFFLE,
) -> list[str]:
    ensure_iris_datasets(DATASETS_DIR)
    csv_files = sorted(DATASETS_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("Nenhum arquivo CSV foi encontrado em ./Bases.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []

    for csv_path in csv_files:
        messages.append(
            process_dataset(
                csv_path=csv_path,
                eta0=eta0,
                max_epochs=max_epochs,
                random_state=random_state,
                shuffle=shuffle,
            )
        )

    return messages


def main() -> None:
    for message in run_all_datasets():
        print(message)


if __name__ == "__main__":
    main()
