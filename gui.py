import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText

from main import ETA0, MAX_EPOCHS, RANDOM_STATE, run_all_datasets


def _parse_inputs(
    eta0_text: str,
    max_epochs_text: str,
    random_state_text: str,
) -> tuple[float, int, int]:
    eta0 = float(eta0_text)
    max_epochs = int(max_epochs_text)
    random_state = int(random_state_text)

    if eta0 <= 0:
        raise ValueError("ETA0 deve ser maior que zero.")
    if max_epochs <= 0:
        raise ValueError("MAX_EPOCHS deve ser maior que zero.")

    return eta0, max_epochs, random_state


def run_gui() -> None:
    window = tk.Tk()
    window.title("Perceptron com Scikit-learn - Lucas Lara")
    window.geometry("800x600")
    window.minsize(800, 600)

    title = tk.Label(
        window,
        text="Execucao do Perceptron",
        font=("Segoe UI", 16, "bold"),
    )
    title.pack(pady=(18, 8))

    description = tk.Label(
        window,
        text="Selecione os parametros para executar todas as bases da pasta Bases.",
        font=("Segoe UI", 10),
    )
    description.pack(pady=(0, 12))

    form = tk.Frame(window, padx=20, pady=10)
    form.pack(fill="x")

    tk.Label(form, text="ETA0", font=("Segoe UI", 10, "bold")).grid(
        row=0, column=0, sticky="w", pady=6
    )
    eta0_entry = tk.Entry(form, width=18)
    eta0_entry.insert(0, str(ETA0))
    eta0_entry.grid(row=0, column=1, sticky="w", padx=(10, 30), pady=6)

    tk.Label(form, text="MAX_EPOCHS", font=("Segoe UI", 10, "bold")).grid(
        row=1, column=0, sticky="w", pady=6
    )
    max_epochs_entry = tk.Entry(form, width=18)
    max_epochs_entry.insert(0, str(MAX_EPOCHS))
    max_epochs_entry.grid(row=1, column=1, sticky="w", padx=(10, 30), pady=6)

    tk.Label(form, text="RANDOM_STATE", font=("Segoe UI", 10, "bold")).grid(
        row=2, column=0, sticky="w", pady=6
    )
    random_state_entry = tk.Entry(form, width=18)
    random_state_entry.insert(0, str(RANDOM_STATE))
    random_state_entry.grid(row=2, column=1, sticky="w", padx=(10, 30), pady=6)

    output = ScrolledText(window, wrap="word", font=("Consolas", 10), height=20)
    output.pack(fill="both", expand=True, padx=20, pady=(8, 20))

    def execute() -> None:
        try:
            eta0, max_epochs, random_state = _parse_inputs(
                eta0_entry.get(),
                max_epochs_entry.get(),
                random_state_entry.get(),
            )
            messages = run_all_datasets(
                eta0=eta0,
                max_epochs=max_epochs,
                random_state=random_state,
            )
        except Exception as exc:
            messagebox.showerror("Erro", str(exc))
            return

        output.delete("1.0", tk.END)
        output.insert(
            tk.END,
            (
                f"ETA0 = {eta0}\n"
                f"MAX_EPOCHS = {max_epochs}\n"
                f"RANDOM_STATE = {random_state}\n\n"
            ),
        )
        output.insert(tk.END, "\n".join(messages))
        output.insert(tk.END, "\n" "Processamento finalizado com sucesso.")
        output.see(tk.END)

    button = tk.Button(
        window,
        text="Executar Bases",
        command=execute,
        font=("Segoe UI", 10, "bold"),
        padx=18,
        pady=8,
    )
    button.pack(pady=(0, 12))

    window.mainloop()


if __name__ == "__main__":
    run_gui()
