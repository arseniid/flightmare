import argparse
import os
import random

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
    model = LearnedMPCShortControlSmall(
        use_batch_normalization=False, activation_fn=Tanh
    )

    hyperparams = {
        "batch_size": 64,
        "learning_rate": 0.0097,
        "lr_decay": 0.0174,
        "momentum": 0.75,
        "optimizer": SGD,
        "loss_fn": SmoothL1Loss(),
        "logger": SummaryWriter("runs/overfit_controls/"),
    }
    train_loader, val_loader = get_dataloaders(
        batch_size=hyperparams["batch_size"], control_only=True
    )

    trainer = Trainer(
        model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        patience=100,
        **hyperparams,
    )
    val_losses = trainer.train(epochs=epochs)

    model_filename = "nmpc_short_controls_model_overfit_hard0.pth"
    trainer.save_model(
        directory="../../models/overfit_controls/", file_name=model_filename
    )
    with open("../../models/overfit_controls/hyperdata.txt", "a+") as f:
        _write_to(
            f_handler=f.write,
            filename=model_filename,
            model_name=model.__class__,
            model_network=model.net,
            **hyperparams,
            final_validation_loss=min(val_losses),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NMPC model")
    parser.add_argument(
        "--hparam-search",
        help="Flag for hyperparameter search",
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
    else:
        train_model(epochs=300)
