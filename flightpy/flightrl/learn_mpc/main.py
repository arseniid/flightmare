import argparse
import os
import random

import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from mpc_dataset import NMPCDataset
from mpc_nn import (ALLOWED_ACTIVATION_FUNCTIONS, LearnedMPCShortControlFirst,
                    LearnedMPCShortControlFirstDeep,
                    LearnedMPCShortControlFirstDeepObstaclesOnly,
                    LearnedMPCShortControlFirstSmall,
                    LearnedMPCShortControlFirstSmallWide, LearnedMPCShortFull,
                    LearnedMPCShortFullSmall)
from torch.nn import (ELU, HuberLoss, L1Loss, LeakyReLU, MSELoss, ReLU,
                      SmoothL1Loss, Tanh)
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils import Trainer


def _write_to(f_handler=None, **kwargs):
    if kwargs["filename"]:
        f_handler(f"\n\n\n=== {kwargs['filename']} ===\n")
    f_handler("Model and hyperparameters are:\n")
    f_handler("\n".join([f"    - {k}: {v}" for k, v in kwargs.items()]))


def _get_controls_only(input, horizon=12, variables=9):
    return input.reshape((horizon - 1, variables))[:, 3:6].flatten()


def _get_controls_first_only(input, horizon=12, variables=9):
    return input.reshape((horizon - 1, variables))[0, 3:6].flatten()


def _get_obstacles_only(input):
    return input[:105]


def get_dataloaders(dataset_folder="nmpc_short", batch_size=8, obstacles_only=False, control_only=False, split=(0.8, 0.2)):
    transform = _get_obstacles_only if obstacles_only else None
    target_transform = _get_controls_first_only if control_only else None

    dataset = NMPCDataset(root_dir=dataset_folder, transform=transform, target_transform=target_transform)
    train_set, val_set = random_split(
        dataset, [round(split[0] * len(dataset)), round(split[1] * len(dataset))]
    )  # 80% / 20% split

    # num_workers should be 1, otherwise there are various zipFile errors due to concurrent read of the same .npz archive
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=1
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=1
    )
    return train_loader, val_loader


def random_hyperparam_search(epochs=10):
    models = [
        LearnedMPCShortFull,
        LearnedMPCShortFullSmall,
        LearnedMPCShortControlFirstSmallWide,
        LearnedMPCShortControlFirstSmall,
        LearnedMPCShortControlFirst,
        LearnedMPCShortControlFirstDeep,
        LearnedMPCShortControlFirstDeepObstaclesOnly,
    ]
    activations = ALLOWED_ACTIVATION_FUNCTIONS
    batch_sizes = [4, 8, 16, 32, 64, 128]
    lr_rates = [1e-5, 1e-2]
    lr_decays = [0, 1e-1]
    momenta = [0, 0.95]
    optimizers = [Adam, SGD]
    loss_fns = [MSELoss, L1Loss, SmoothL1Loss, HuberLoss]

    best_val_loss = 1e5
    for i in range(30):
        cool_nl = "\n" * bool(i)  # That's so cool!
        print(f"{cool_nl}=== Random search #{i} starts ===")

        use_batch_norm = bool(random.getrandbits(1))  # chooses True/False randomly
        if use_batch_norm:  # BatchNorm is recommended to use with batch_size >= 32
            batch_sizes = [16, 32, 64, 128]
        activation = random.choice(activations)

        model = random.choice(models[2:])(  # TODO: Manage models choice here!
            use_batch_normalization=use_batch_norm, activation_fn=activation
        )  # initialized!

        obstacles = len(list(filter(lambda x: isinstance(model, x), models[-1:]))) != 0
        control = len(list(filter(lambda x: isinstance(model, x), models[2:]))) != 0

        hyperparams = {
            "batch_size": random.choice(batch_sizes),
            "learning_rate": random.uniform(*lr_rates),
            "lr_decay": random.uniform(*lr_decays),
            "momentum": random.uniform(*momenta),
            "optimizer": random.choice(optimizers),
            "loss_fn": random.choice(loss_fns)(),  # initialized!
            "logger": SummaryWriter(log_dir=f"runs/hparam_search/exp{i}"),
        }
        train_loader, val_loader = get_dataloaders(
            batch_size=hyperparams["batch_size"],
            obstacles_only=obstacles,
            control_only=control,
        )

        trainer = Trainer(
            model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            patience=5,
            **hyperparams,
        )
        val_losses = trainer.train(epochs=epochs)

        val_losses_better = [x <= best_val_loss for x in val_losses]
        val_loss_okay = [x <= 1.3 for x in val_losses]
        if any(val_losses_better + val_loss_okay):
            print(f"Current best val loss: {best_val_loss}")
            print(f"Current val losses: {val_losses}")
            best_val_loss = min(val_losses)
            _write_to(
                f_handler=print,
                filename=None,
                model_name=model.__class__,
                model_network=model.net,
                **hyperparams,
            )


def train_model(epochs):
    model = LearnedMPCShortControlFirstDeepObstaclesOnly(
        use_batch_normalization=False, activation_fn=Tanh
    )

    hyperparams = {
        "batch_size": 16,
        "learning_rate": 0.0035,
        "lr_decay": 0.047,
        "momentum": 0.14,
        "optimizer": SGD,
        "loss_fn": SmoothL1Loss(),
        "logger": SummaryWriter("runs/obstacles_only/"),
    }
    train_loader, val_loader = get_dataloaders(
        batch_size=hyperparams["batch_size"],
        obstacles_only=True,
        control_only=True,
    )

    trainer = Trainer(
        model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        patience=30,
        **hyperparams,
    )
    val_losses = trainer.train(epochs=epochs)

    model_filename = "nmpc_short_controls_first_model_deep_obstacles_only.pth"
    trainer.save_model(
        directory="../../models/obstacles_only/", file_name=model_filename
    )
    with open("../../models/obstacles_only/hyperdata.txt", "a+") as f:
        _write_to(
            f_handler=f.write,
            filename=model_filename,
            model_name=model.__class__,
            model_network=model.net,
            **hyperparams,
            final_validation_loss=min(val_losses),
        )


def plot_feature_importance(manual=True):
    train_loader, _ = get_dataloaders(
        batch_size=16945,
        obstacles_only=False,
        control_only=True,
        split=(1.0, 0.0),
    )

    # Split data into X and y
    X, y = next(iter(train_loader))

    label = {
        0: "x",
        1: "y",
        2: "z"
    }

    if manual:
        fig, ax = plt.subplots()
        for i in range(3):
            # Fit model to training data
            model = xgb.XGBRegressor()
            model.fit(X, y[:, i])

            bar_width = 0.3
            x = (-bar_width if i == 0 else +bar_width) if (i % 2 == 0) else 0
            _ = ax.bar(np.arange(len(model.feature_importances_)) + x, model.feature_importances_, bar_width, label=f"{label[i]}-coordinate")

            # Add title and custom x-axis tick labels
            ax.set_ylabel("Feature importance")
            ax.set_title("Feature importances by coordinates")
            ax.tick_params(axis='x', rotation=60)
            ax.set_xticks(np.arange(len(model.feature_importances_)))
            ax.legend()
        plt.show()
    else:
        for i in range(3):
            model = xgb.XGBRegressor()
            model.fit(X, y[:, i])

            xgb.plot_importance(model)
            plt.title(f"Feature importance for output velocity in {label[i]}-coordinate")
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NMPC model")
    parser.add_argument(
        "--hparam-search",
        help="Flag for hyperparameter search",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--feature-importance",
        help="Flag for plotting the feature importances",
        required=False,
        action="store_true",
    )
    args = parser.parse_args()

    if os.getcwd().split("/")[-1] != "learn_mpc":
        raise FileNotFoundError(
            "Saving of the trained model depends on the current working directory. "
            "Please run the script from the 'agile_flight/flightmare/flightpy/flightrl/learn_mpc/' folder!"
        )

    if args.hparam_search:
        print(random_hyperparam_search())
    elif args.feature_importance:
        plot_feature_importance()
    else:
        train_model(epochs=100)
