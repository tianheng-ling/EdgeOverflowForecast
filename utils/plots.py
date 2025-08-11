import os

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def plot_learning_curve(
    epochs: list, train_losses: list, val_losses: list, save_path: str, prefix: bool
):
    prefix = "QAT_" if prefix else ""
    save_path = os.path.join(save_path, f"{prefix}learning_curve.png")

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, train_losses, label=f"Train Loss")
    ax.plot(epochs, val_losses, label=f"Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close()


def plot_preds_truths(
    preds: list, truths: list, plot_len: int, save_path: str, prefix=None
):
    plt.plot(range(plot_len), truths[:plot_len], color="green", label="Ground Truth")
    plt.plot(range(plot_len), preds[:plot_len], color="red", label="Prediction")
    plt.ylabel("Filling Level(m)")
    plt.xlabel("Time Steps")
    plt.legend()
    plt.savefig(str(save_path) + f"/{prefix}preds_truths.pdf", dpi=300, format="pdf")
    plt.clf()
    plt.close()


def plot_pareto(
    losses, hw_metrics, non_pareto_losses, non_pareto_hw_metrics, save_path: str
):
    plt.figure(figsize=(4, 3))

    plt.scatter(
        non_pareto_losses,
        non_pareto_hw_metrics,
        c="blue",
        label="Non-Pareto Front",
        alpha=0.6,
    )

    plt.scatter(losses, hw_metrics, c="red", label="Pareto Front", alpha=0.8)

    plt.xlabel("Val MSE")
    plt.ylabel("Energy (mW)")
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
