from torch import nn

ALLOWED_ACTIVATION_FUNCTIONS = [
    nn.ReLU,
    nn.ELU,
    nn.LeakyReLU,
    nn.Tanh,
]


class BaseLearnedMPC(nn.Module):
    def __init__(self, batch_norm=None, activation=None) -> None:
        super().__init__()
        self.flatten = nn.Flatten()

        self.batch_norm_layer = nn.BatchNorm1d if batch_norm else nn.Identity

        assert activation in ALLOWED_ACTIVATION_FUNCTIONS
        self.activation = activation

        self.net = None

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits

    def _block(self, in_size, out_size):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            self.batch_norm_layer(out_size),
            self.activation(),
        )

    def _final_block(self, in_size, out_size):
        return nn.Linear(in_size, out_size)


class LearnedMPCShortFull(BaseLearnedMPC):
    """
    Naming:
        - LearnedMPC -> learning-based MPC
        - Short -> trained on the short horizon (i.e., of 12 steps)
        - Full -> predicts full MPC output (not only controls)
    """
    def __init__(self, use_batch_normalization, activation_fn=nn.ReLU):
        super().__init__(batch_norm=use_batch_normalization, activation=activation_fn)

        self.net = nn.Sequential(
            # in: 7-dim. relative state of 15 obstacles + 3-dim. goal direction (in drone body frame) = 108
            self._block(in_size=15 * 7 + 3, out_size=512),
            self._block(512, 1024),
            self._block(1024, 512),
            self._block(512, 256),
            self._final_block(in_size=256, out_size=11 * 9),
            # out: horizon (T-1) of 11 x 9-dim. predicted NMPC output = 99
        )


class LearnedMPCShortFullSmall(BaseLearnedMPC):
    """
    Naming:
        - LearnedMPC -> learning-based MPC
        - Short -> trained on the short horizon (i.e., of 12 steps)
        - Full -> predicts full MPC output (not only controls)
        - Small -> size of the network, i.e., 4 linear layers only
    """
    def __init__(self, use_batch_normalization, activation_fn=nn.ReLU):
        super().__init__(batch_norm=use_batch_normalization, activation=activation_fn)

        self.net = nn.Sequential(
            # in: 7-dim. relative state of 15 obstacles + 3-dim. goal direction (in drone body frame) = 108
            self._block(in_size=15 * 7 + 3, out_size=256),
            self._block(256, 512),
            self._block(512, 256),
            self._final_block(in_size=256, out_size=11 * 9),
            # out: horizon (T-1) of 11 x 9-dim. predicted NMPC output = 99
        )


class LearnedMPCShortControlSmall(BaseLearnedMPC):
    """
    Naming:
        - LearnedMPC -> learning-based MPC
        - Short -> trained on the short horizon (i.e., of 12 steps)
        - Control -> predicts only control part of the MPC output (i.e., velocities)
        - Small -> size of the network, i.e., 3 (!) linear layers only
    """
    def __init__(self, use_batch_normalization, activation_fn=nn.ReLU):
        super().__init__(batch_norm=use_batch_normalization, activation=activation_fn)

        self.net = nn.Sequential(
            # in: 7-dim. relative state of 15 obstacles + 3-dim. goal direction (in drone body frame) = 108
            self._block(in_size=15 * 7 + 3, out_size=256),
            self._block(256, 128),
            self._final_block(in_size=128, out_size=11 * 3),
            # out: horizon (T-1) of 11 x 3-dim. drone controls = 33
        )
