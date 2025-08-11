import os
import json
import wandb
import optuna
import argparse
from functools import partial
from optuna.storages import RDBStorage
from optuna.samplers import NSGAIISampler, TPESampler

from train_val import train_val
from test import test
from data.TimeSeriesDataset import TimeSeriesDataset
from config import (
    data_default_config,
    rnn_default_config,
    transformer_default_config,
)
from utils.set_paths import set_base_paths
from utils.plots import plot_pareto
from hw_converter.convert2hw import convert2hw
from optuna_utils import vivado_runner
from config import search_space


def objective(trial, args):
    try:

        is_qat = args.is_qat
        model_type = args.model_type

        # common search space
        common_space = search_space["common"]
        quant_bits = (
            trial.suggest_int("quant_bits", **common_space["quant_bits"])
            if is_qat
            else None
        )
        batch_size = trial.suggest_int("batch_size", **common_space["batch_size"])
        lr = trial.suggest_float("lr", **common_space["lr"])

        model_space = search_space.get(model_type)
        if model_space is None:
            raise ValueError(f"Unsupported model_type: {model_type}")

        if model_type == "lstm":
            hidden_size = trial.suggest_int("hidden_size", **model_space["hidden_size"])
        elif model_type == "transformer":
            d_model = trial.suggest_int("d_model", **model_space["d_model"])

        # Get datasets
        data_config = data_default_config.copy()
        data_config["window_size"] = args.window_size
        train_dataset = TimeSeriesDataset(mode="train", data_config=data_config)
        val_dataset = TimeSeriesDataset(mode="val", data_config=data_config)
        test_dataset = TimeSeriesDataset(mode="test", data_config=data_config)

        # set model configs
        if model_type == "lstm":
            model_params = rnn_default_config.copy()
            model_params["hidden_size"] = hidden_size
        elif model_type == "transformer":
            model_params = transformer_default_config.copy()
            model_params["d_model"] = d_model
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        model_params.update(
            {
                "batch_size": batch_size,
                "model_type": model_type,
                "num_in_features": 1,
                "num_out_features": 1,
                "window_size": args.window_size,
                "is_qat": is_qat,
            }
        )
        if is_qat:
            model_params.update(
                {
                    "name": "network",
                    "quant_bits": quant_bits,
                    "do_int_forward": False,
                }
            )

        # set exp_save_path
        exp_base_save_dir = args.exp_base_save_dir
        given_timestamp = ""
        exp_save_dir, fig_save_dir, log_save_dir = set_base_paths(
            "train", exp_base_save_dir, given_timestamp
        )
        trial.set_user_attr("timestamp", str(exp_save_dir).split("/")[-1])

        # set up wandb
        wandb.init(
            project=args.wandb_project_name,
            mode=args.wandb_mode,
            config=args,
            name=f"trial_{trial.number}",
        )
        wandb.log(
            {
                "batch_size": batch_size,
                "lr": lr,
            }
        )

        # execute training and validation
        best_val_loss, _, _, _ = train_val(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_params=model_params,
            batch_size=batch_size,
            lr=lr,
            num_epochs=args.num_epochs,
            exp_save_dir=exp_save_dir,
            fig_save_dir=fig_save_dir,
            log_save_dir=log_save_dir,
        )

        # execute test
        test_config = {
            "test_dataset": test_dataset,
            "model_params": model_params,
            "exp_mode": "train",
            "batch_size": batch_size,
            "exp_save_dir": exp_save_dir,
            "fig_save_dir": fig_save_dir,
            "log_save_dir": log_save_dir,
        }
        test_loss = test(**test_config)
        trial.set_user_attr("test_loss", test_loss if test_loss else None)

        if is_qat:
            # integer-only inference
            model_params["do_int_forward"] = True
            int_test_loss = test(**test_config)
            trial.set_user_attr(
                "int_test_loss", int_test_loss if int_test_loss else None
            )

            # convert to target and harware simulation
            if args.do_hw_simulation:

                # convert to target hw
                convert2hw(
                    model_type=model_type,
                    test_dataset=test_dataset,
                    subset_size=args.subset_size,
                    model_params=model_params,
                    exp_save_dir=exp_save_dir,
                )
                # get vivado report info
                hw_metrics = vivado_runner(
                    base_dir=os.path.abspath(os.path.join(exp_save_dir, "hw")),
                    top_module="network",
                    trial=trial,
                )

                power = hw_metrics["power"]
                latency = hw_metrics["latency"]
                energy = hw_metrics["energy"]
                trial.set_user_attr(
                    "power",
                    round(float(power), 3) if power is not None else None,
                )
                trial.set_user_attr(
                    "latency",
                    round(float(latency), 3) if latency is not None else None,
                )
                trial.set_user_attr(
                    "energy",
                    round(float(energy), 3) if energy is not None else None,
                )
        trial.set_user_attr(
            "best_val_loss",
            round(float(best_val_loss), 3) if best_val_loss is not None else None,
        )

        return best_val_loss, hw_metrics[args.optuna_hw_target]
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()
    finally:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # wandb configs
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument(
        "--wandb_mode", type=str, choices=["online", "offline", "disabled"]
    )
    # data configs
    parser.add_argument("--window_size", type=int)
    # model configs
    parser.add_argument("--model_type", type=str, choices=["lstm", "transformer"])
    parser.add_argument("--is_qat", action="store_true", help="normal training or qat")
    # exp configs
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--exp_base_save_dir", type=str)

    # hw conversion configs
    parser.add_argument(
        "--subset_size",
        type=int,
        help="Number of samples to use for HW simulation",
    )
    parser.add_argument(
        "--do_hw_simulation",
        action="store_true",
        help="Do HW simulation for quantized model",
    )
    # optuna configs
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument(
        "--optuna_hw_target",
        type=str,
        choices=["power", "latency", "energy"],
    )
    args = parser.parse_args()

    os.makedirs(args.exp_base_save_dir, exist_ok=True)
    db_name = f"{args.model_type}_{args.is_qat}.db"
    db_path = os.path.join(args.exp_base_save_dir, db_name)
    storage = RDBStorage(f"sqlite:///{db_path}")

    study = optuna.create_study(
        directions=["minimize", "minimize"],  # [val_loss, hw_metric]
        sampler=TPESampler(),  # NSGAIISampler(),  # (seed=seed),
        storage=storage,
        load_if_exists=True,
        study_name=f"{args.model_type}_{args.is_qat}",
    )
    study.optimize(
        partial(objective, args=args), n_trials=args.n_trials, catch=(Exception,)
    )

    # Save the all trial to JSON
    with open(
        f"{args.exp_base_save_dir}/all_trials_{args.model_type}_{args.is_qat}.json",
        "w",
    ) as f:
        all_trials_data = [
            {
                "trial": t.number,
                "val_loss": (
                    round(t.values[0], 3)
                    if t.values is not None and t.values[0] is not None
                    else None
                ),
                "hw_metric": (
                    round(t.values[1], 3)
                    if t.values is not None and t.values[1] is not None
                    else None
                ),
                "params": t.params,
                "user_attrs": t.user_attrs,
                "state": t.state.name,  # Save the state (e.g., COMPLETE, FAIL)
            }
            for t in study.trials
        ]
        json.dump(all_trials_data, f, indent=4)

    # Save Pareto front trials to JSON
    with open(
        f"{args.exp_base_save_dir}/pareto_trials_{args.model_type}_{args.is_qat}.json",
        "w",
    ) as f:
        pareto_data = [
            {
                "trial": t.number,
                "val_loss": (
                    round(t.values[0], 3)
                    if t.values is not None and t.values[0] is not None
                    else None
                ),
                "hw_metric": (
                    round(t.values[1], 3)
                    if t.values is not None and t.values[1] is not None
                    else None
                ),
                "params": t.params,
                "user_attrs": t.user_attrs,
            }
            for t in study.best_trials
        ]
        json.dump(pareto_data, f, indent=4)

    # Plot Pareto front
    all_trials = study.trials
    pareto_trials = study.best_trials

    pareto_losses = []
    pareto_hw_metrics = []
    for t in pareto_trials:
        if t.values is not None and t.values[0] is not None and t.values[1] is not None:
            pareto_losses.append(t.values[0])
            pareto_hw_metrics.append(t.values[1])

    non_pareto_losses = []
    non_pareto_hw_metrics = []
    for t in all_trials:
        if (
            t not in pareto_trials
            and t.values is not None
            and t.values[0] is not None
            and t.values[1] is not None
        ):
            non_pareto_losses.append(t.values[0])
            non_pareto_hw_metrics.append(t.values[1])

    plot_pareto(
        pareto_losses,
        pareto_hw_metrics,
        non_pareto_losses,
        non_pareto_hw_metrics,
        save_path=f"{args.exp_base_save_dir}/pareto_plot_{args.model_type}_{args.is_qat}.pdf",
    )
