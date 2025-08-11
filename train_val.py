import os
import torch
import wandb
import logging
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader

from config import DEVICE
from models.build_model import build_model
from utils.set_logger import setup_logger
from utils.EarlyStopping import EarlyStopping
from utils.plots import plot_learning_curve
from utils.get_model_complexity import analyze_model_memory


def train_val(
    model_params: dict,
    batch_size: int,
    lr: float,
    num_epochs: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
    exp_save_dir: str,
    fig_save_dir: str,
    log_save_dir: str,
) -> object:

    # set up data loaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    # build model
    model = build_model(model_params).to(DEVICE)
    wandb.log({"timestemp": str(exp_save_dir).split("/")[-1], **model_params})
    # select criterion
    criterion = torch.nn.MSELoss()

    # set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # set up early stopping
    early_stopping = EarlyStopping(
        patience=10,
        verbose=True,
        delta=0,
        path=exp_save_dir,
        trace_func=print,
    )

    # set up logging
    logger = setup_logger(
        "train_val_logger", os.path.join(log_save_dir, "train_val_logfile.log")
    )

    # execute train and validation phase
    all_train_epoch_losses = []
    all_val_epoch_losses = []

    for epoch in range(num_epochs):

        # train phase
        model.train()
        sum_train_batch_losses = 0
        for (
            train_samples,
            train_target,
        ) in train_dataloader:

            train_samples = train_samples.to(DEVICE)
            train_target = train_target.to(DEVICE)
            train_pred = model(inputs=train_samples)
            train_batch_loss = criterion(train_pred, train_target)
            optimizer.zero_grad()
            train_batch_loss.backward()
            optimizer.step()
            sum_train_batch_losses += train_batch_loss.item()
        train_epoch_loss = sum_train_batch_losses / len(train_dataloader)
        all_train_epoch_losses.append(train_epoch_loss)

        # validation phase
        model.eval()
        sum_val_batch_losses = 0
        with torch.no_grad():
            for val_samples, val_target in val_dataloader:
                val_samples = val_samples.to(DEVICE)
                val_target = val_target.to(DEVICE)
                val_pred = model(inputs=val_samples)
                val_batch_loss = criterion(val_pred, val_target)
                sum_val_batch_losses += val_batch_loss.item()
            val_epoch_loss = sum_val_batch_losses / len(val_dataloader)
            all_val_epoch_losses.append(val_epoch_loss)

        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            logging.info("executed_valid_epochs: {}".format(epoch - 10))
            break

        # Log results
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_epoch_loss,
            "val_loss": val_epoch_loss,
        }
        headers = ["epoch", "train_loss", "val_loss"]
        row = [[epoch + 1, f"{train_epoch_loss:.4f}", f"{val_epoch_loss:.4f}"]]
        logger.info(tabulate(row, headers=headers, tablefmt="pretty"))
        wandb.log(metrics)

    plot_learning_curve(
        epochs=range(1, len(all_train_epoch_losses) + 1),
        train_losses=all_train_epoch_losses,
        val_losses=all_val_epoch_losses,
        save_path=fig_save_dir,
        prefix=model_params["is_qat"],
    )

    best_val_loss = min(all_val_epoch_losses)
    weights_mem, activations_mem, total_macs = analyze_model_memory(
        model_params=model_params, model_type=model_params["model_type"], model=model
    )
    wandb.log(
        {
            "best_val_loss": best_val_loss,
            "weights_mem": weights_mem,
            "activations_mem": activations_mem,
            "total_macs": total_macs,
        }
    )
    return best_val_loss, weights_mem, activations_mem, total_macs
