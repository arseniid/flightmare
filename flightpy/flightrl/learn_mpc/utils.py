import os

import numpy as np
import torch


class DataSaver:
    def __init__(self, folder) -> None:
        subpkg = "envtest"
        cwd = os.getcwd()
        self.dir = (
            cwd[: cwd.find(subpkg)] + "flightmare/flightpy/datasets/" + folder + "/"
        )

        self.metadata = self._read_metadata()

    def save_data(self, input_data, output_data, **kwargs):
        environment = kwargs["environment"]

        update_metadata = False
        for idx, data in enumerate([input_data, output_data]):
            file_to_save_path = (
                f"{self.dir}{'input' if idx == 0 else 'output'}_{environment}.npy"
            )
            if not os.path.exists(file_to_save_path):
                np.save(file_to_save_path, data)
                update_metadata = True
            elif kwargs["crashes"] < self.metadata[environment]:
                np.save(file_to_save_path, data)
                update_metadata = True
        if update_metadata:
            self._add_metadata(**kwargs)

    def _read_metadata(self):
        metadata = dict()
        with open(self.dir + "metadata.txt", "r+") as metadata_file:
            metadata_lines = metadata_file.read().splitlines()
            for line in metadata_lines:
                env = line.split()[2].rstrip(":")
                collisions = int(line.split()[-2])
                metadata[env] = collisions
        return metadata

    def _add_metadata(self, **kwargs):
        with open(self.dir + "metadata.txt", "a+") as metadata_file:
            metadata_file.write(
                f"\nDataset from {kwargs['environment']}: Stored {kwargs['size']} data sequences with overall {kwargs['crashes']} collisions"
            )
        self.metadata[kwargs["environment"]] = kwargs["crashes"]


class Trainer:
    """
    Universal class to train PyTorch neural networks.

    Some (important) attributes
    ----------
    model : instance of PyTorch nn.Module (or derived) class
        model to train
        [must be already initialized!]
    opt : torch.optim.Optimizer (or derived class)
        optimization algorithm
        [will be initialized here - no need to pass initialized argument!]
    loss_fn : instance of torch.nn._Loss (or derived) class
        function to compute loss between prediction and labels
        [must be already initialized!]
    writer : instance of tensorboard.SummaryWriter class
        Tensorboard logs writer
        [must be already initialized!]
    """

    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        batch_size=8,
        learning_rate=1e-3,
        lr_decay=1e-2,
        momentum=0.9,
        optimizer=None,
        loss_fn=None,
        verbose=True,
        print_every=100,
        logger=None,
        **kwargs,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.model.to(self.device)

        self.loss_fn = loss_fn

        try:
            self.opt = optimizer(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=lr_decay,
                momentum=momentum,
            )
        except TypeError as e:
            print(
                f"{e}: Adam optimizer doesn't use momentum! Initializing Adam without it."
            )
            self.opt = optimizer(
                self.model.parameters(), lr=learning_rate, weight_decay=lr_decay
            )

        self.verbose = verbose
        self.print_every = print_every
        self.writer = logger

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.current_patience = 0

    def train(self, epochs):
        val_losses = []
        for epoch in range(epochs):
            size = len(self.train_dataloader.dataset)
            self.model.train()
            for batch, (X, y) in enumerate(self.train_dataloader):
                X, y = X.float().to(self.device), y.float().to(self.device)

                # Compute prediction error
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                self.writer.add_scalar("Loss/train", loss, epoch)

                # Backpropagation
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if self.verbose and batch % self.print_every == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            val_losses.append(self.val(epoch))

        self.writer.flush()  # make sure that all pending events have been written to disk
        self.writer.close()
        return val_losses

    def val(self, epoch):
        num_batches = len(self.val_dataloader)
        self.model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.val_dataloader:
                X, y = X.float().to(self.device), y.float().to(self.device)
                pred = self.model(X)
                val_loss += self.loss_fn(pred, y).item()
        val_loss /= num_batches
        self.writer.add_scalar("Loss/val", val_loss, epoch)
        # TODO: Report number of collisions (instead of accuracy)
        if self.verbose:
            print(
                f"Val Error after epoch #{epoch}: \n Accuracy: {(100*correct):>0.1f}%, Avg. loss: {val_loss:>8f} \n"
            )
        return val_loss

    def save_model(self, directory="models", file_name="model.pth"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_path = os.path.join(directory, file_name)
        model = self.model.cpu()
        torch.save(model.state_dict(), model_path)
        print(f"The model is saved to {model_path} successfully!")
        return model_path
