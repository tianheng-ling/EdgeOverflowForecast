import os
import torch
import wandb
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader

from config import DEVICE
from models.build_model import build_model
from utils.set_logger import setup_logger
from utils.plots import plot_preds_truths
from utils.get_model_complexity import analyze_model_memory


def test(
    test_dataset: Dataset,
    batch_size: int,
    model_params: dict,
    exp_mode: str,
    exp_save_dir: str,
    fig_save_dir: str,
    log_save_dir: str,
) -> object:

    # get test data loader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    # Load model from the best checkpoint
    model = build_model(model_params).to(DEVICE)
    checkpoint = torch.load(exp_save_dir / "best_model.pth", weights_only=True)
    model.load_state_dict(checkpoint, strict=False)

    # Set up logging
    logger = setup_logger("test_logger", os.path.join(log_save_dir, "test_logfile.log"))

    # Set criterion
    criterion = torch.nn.MSELoss()

    # set prefix
    prefix = (
        "int_"
        if model_params["is_qat"] == True and model_params["do_int_forward"] == True
        else ""
    )

    # calculate model size in KB
    model_complexity = analyze_model_memory(
        model_params=model_params, model_type=model_params["model_type"], model=model
    )
    model_complexity = {
        "weights_size(KB)": model_complexity[0],
        "activations_size(KB)": model_complexity[1],
        "total_macs": model_complexity[2],
    }

    # perform test
    model.eval()
    sum_test_batch_losses = 0
    sum_test_batch_losses_denorm = 0
    all_test_targets = []
    all_test_preds = []
    with torch.no_grad():
        for idx, (test_samples, test_target) in enumerate(test_dataloader):

            test_samples = test_samples.to(DEVICE)
            test_target = test_target.to(DEVICE)
            test_pred = model(inputs=test_samples)
            test_pred = test_pred.to(test_target.device)
            test_batch_loss = criterion(test_pred, test_target)
            test_pred_denorm = test_dataset.inverse_transform(
                test_pred.view(-1, 1).cpu().detach().numpy()
            )
            test_target_denorm = test_dataset.inverse_transform(
                test_target.view(-1, 1).cpu().detach().numpy()
            )

            test_batch_loss_denorm = criterion(
                torch.tensor(test_pred_denorm, dtype=torch.float32),
                torch.tensor(test_target_denorm, dtype=torch.float32),
            )

            sum_test_batch_losses += test_batch_loss.item()
            sum_test_batch_losses_denorm += test_batch_loss_denorm.item()

            all_test_targets.append(test_target_denorm)
            all_test_preds.append(test_pred_denorm)

    test_loss = sum_test_batch_losses / len(test_dataloader)
    test_loss_denorm = sum_test_batch_losses_denorm / len(test_dataloader)

    metrics = {
        f"{prefix}test_loss": f"{test_loss:.4f}",
        f"{prefix}test_loss_denorm": f"{test_loss_denorm:.4f}",
    }

    # Print results
    print(f"---------------- {prefix}Test Results ----------------")
    headers = list(metrics.keys()) + list(model_complexity.keys())
    row = [
        [metrics[k] for k in metrics.keys()]
        + [
            (
                model_complexity[k]
                if isinstance(model_complexity[k], (int, float))
                else str(model_complexity[k])
            )
            for k in model_complexity.keys()
        ]
    ]
    logger.info(tabulate(row, headers=headers, tablefmt="pretty"))
    if exp_mode == "train":
        wandb.log({**metrics, **model_complexity})

    plot_preds_truths(
        preds=all_test_preds[0],
        truths=all_test_targets[0],
        save_path=fig_save_dir,
        plot_len=batch_size,
        prefix=prefix,
    )

    return test_loss_denorm
