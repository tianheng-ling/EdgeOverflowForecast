import wandb
import argparse

from data.TimeSeriesDataset import TimeSeriesDataset
from config import (
    data_default_config,
    rnn_default_config,
    transformer_default_config,
)
from utils.set_paths import set_base_paths
from train_val import train_val
from test import test
from hw_converter.convert2hw import convert2hw


def main(args):

    # Get datasets
    data_config = data_default_config.copy()
    data_config.update(
        {
            "window_size": args.window_size,
        }
    )
    train_dataset = TimeSeriesDataset(mode="train", data_config=data_config)
    val_dataset = TimeSeriesDataset(mode="val", data_config=data_config)
    test_dataset = TimeSeriesDataset(mode="test", data_config=data_config)

    # set exp_save_path
    exp_save_dir, fig_save_dir, log_save_dir = set_base_paths(
        args.exp_mode, args.exp_base_save_dir, args.given_timestamp
    )

    # set model configs
    if args.model_type == "lstm":
        model_params = rnn_default_config.copy()
    elif args.model_type == "transformer":
        model_params = transformer_default_config.copy()
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")
    model_params.update(
        {
            "batch_size": args.batch_size,
            "model_type": args.model_type,
            "num_in_features": 1,
            "num_out_features": 1,
            "window_size": args.window_size,
            "is_qat": args.is_qat,
        }
    )

    if args.is_qat:
        model_params.update(
            {
                "name": "network",
                "quant_bits": args.quant_bits,
                "do_int_forward": False,
            }
        )

    if args.exp_mode == "train":
        # set up wandb
        wandb.init(project=args.wandb_project_name, mode=args.wandb_mode, config=args)
        train_val(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_params=model_params,
            batch_size=args.batch_size,
            lr=args.lr,
            num_epochs=args.num_epochs,
            exp_save_dir=exp_save_dir,
            fig_save_dir=fig_save_dir,
            log_save_dir=log_save_dir,
        )

    test_config = {
        "test_dataset": test_dataset,
        "model_params": model_params,
        "exp_mode": args.exp_mode,
        "batch_size": args.batch_size,
        "exp_save_dir": exp_save_dir,
        "fig_save_dir": fig_save_dir,
        "log_save_dir": log_save_dir,
    }

    test(**test_config)
    if args.is_qat:
        # integer-only inference
        model_params["do_int_forward"] = True
        test(**test_config)
        # convert to target hw
        convert2hw(
            model_type=args.model_type,
            test_dataset=test_dataset,
            subset_size=args.subset_size,
            model_params=model_params,
            exp_save_dir=exp_save_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting")

    # wandb configs
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument(
        "--wandb_mode", type=str, choices=["online", "offline", "disabled"]
    )

    # data configs
    parser.add_argument("--window_size", type=int)

    # exp configs
    parser.add_argument("--exp_mode", type=str, choices=["train", "test"])
    parser.add_argument("--given_timestamp", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--exp_base_save_dir", type=str)

    # model configs
    parser.add_argument("--model_type", type=str, choices=["lstm", "transformer"])
    parser.add_argument("--is_qat", action="store_true", help="normal training or qat")

    # quantization configs
    parser.add_argument("--quant_bits", type=int, choices=[4, 6, 8])

    # hw conversion configs
    parser.add_argument(
        "--subset_size",
        type=int,
        help="Number of samples to use for HW simulation",
    )

    args = parser.parse_args()

    main(args)
