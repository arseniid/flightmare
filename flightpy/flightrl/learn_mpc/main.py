import argparse
import os
import random

from torch.nn import ELU, HuberLoss, LeakyReLU, MSELoss, ReLU, Tanh
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from mpc_dataset import NMPCDataset
from mpc_nn import (ALLOWED_ACTIVATION_FUNCTIONS, MPCLearnedControl,
                    MPCLearnedControlSmall, MPCLearnedFullSmall)
from utils import Trainer


def _write_to(f_handler=None, **kwargs):
    if kwargs["filename"]:
        f_handler(f"=== {kwargs['filename']} ===")
    f_handler("Model and hyperparameters are:")
    f_handler("\n".join([f"    - {k}: {v}" for k, v in kwargs.items()]))


def get_dataloaders(dataset_folder="nmpc", batch_size=8):
    dataset = NMPCDataset(root_dir=dataset_folder)
    train_set, val_set = random_split(
        dataset, [round(0.8 * len(dataset)), round(0.2 * len(dataset))]
    )  # 80% / 20% split

    # num_workers should be 1, otherwise there are various zipFile errors due to concurrent read of the same .npz archive
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=1
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=1
    )
    return train_loader, val_loader


def random_hyperparam_search(epochs=5):
    models = [MPCLearnedControl, MPCLearnedControlSmall]
    activations = ALLOWED_ACTIVATION_FUNCTIONS
    batch_sizes = [4, 8, 16, 32, 64, 128]
    lr_rates = [1e-5, 1e-2]
    lr_decays = [0, 1e-1]
    momenta = [0, 0.9]
    optimizers = [Adam, SGD]
    loss_fns = [MSELoss, HuberLoss]

    best_val_loss = 1e5
    for i in range(150):
        print(f"{'\n' * bool(i)}=== Random search #{i} starts ===")  # That's so cool!

        use_batch_norm = bool(random.getrandbits(1))  # chooses True/False randomly
        if use_batch_norm:  # BatchNorm is recommended to use with batch_size >= 32
            batch_sizes = [16, 32, 64, 128]
        activation = random.choice(activations)

        model = random.choice(models)(
            use_batch_normalization=use_batch_norm, activation_fn=activation
        )  # initialized!

        hyperparams = {
            "batch_size": random.choice(batch_sizes),
            "learning_rate": random.uniform(*lr_rates),
            "lr_decay": random.uniform(*lr_decays),
            "momentum": random.uniform(*momenta),
            "optimizer": random.choice(optimizers),
            "loss_fn": random.choice(loss_fns)(),  # initialized!
            "logger": SummaryWriter(log_dir=f"runs/hparam_search/exp{i}"),
        }
        train_loader, val_loader = get_dataloaders(batch_size=hyperparams["batch_size"])

        trainer = Trainer(
            model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            **hyperparams,
        )
        val_losses = trainer.train(epochs=epochs)

        val_losses_better = [x <= best_val_loss for x in val_losses]
        val_loss_okay = [x <= 1.0 for x in val_losses]
        if any(val_losses_better + val_loss_okay):
            print(f"Current best val loss: {best_val_loss}")
            print(f"Current val losses: {val_losses}")
            best_val_loss = min(val_losses)
            _write_to(
                f_handler=print, filename=None, model_network=model.net, **hyperparams,
            )


def train_model(epochs):
    model = MPCLearnedControl(use_batch_normalization=True)

    hyperparams = {
        "batch_size": 64,
        "learning_rate": 7e-3,
        "lr_decay": 3e-3,
        "momentum": 0.0,
        "optimizer": Adam,
        "loss_fn": MSELoss(),
        "logger": SummaryWriter(),
    }
    train_loader, val_loader = get_dataloaders(batch_size=hyperparams["batch_size"])

    trainer = Trainer(
        model, train_dataloader=train_loader, val_dataloader=val_loader, **hyperparams,
    )
    trainer.train(epochs=epochs)

    model_filename = "nmpc_control_model.pth"
    trainer.save_model(directory="../../models", file_name=model_filename)
    with open("../../models/hyperdata.txt", "a+") as f:
        _write_to(
            f_handler=f.write,
            filename=model_filename,
            model_network=model.net,
            **hyperparams,
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
        train_model(epochs=20)
