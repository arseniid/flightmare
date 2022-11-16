from torch import nn

ALLOWED_ACTIVATION_FUNCTIONS = [
    nn.ReLU,
    nn.ELU,
    nn.LeakyReLU,
    nn.Tanh,
]


class BaseMPCLearned(nn.Module):
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


class MPCLearnedControl(BaseMPCLearned):
    def __init__(self, use_batch_normalization, activation_fn=nn.ReLU):
        super().__init__(batch_norm=use_batch_normalization, activation=activation_fn)

        self.net = nn.Sequential(
            # in: 7-dim. state of 15 obstacles + 6-dim. drone state = 111
            self._block(in_size=15 * 7 + 6, out_size=512),
            self._block(512, 1024),
            self._block(1024, 512),
            self._block(512, 256),
            self._final_block(in_size=256, out_size=3 * 24),
            # out: 3-dim. drone controls x horizon (T-1) of 24 = 72
        )


class MPCLearnedControlSmall(BaseMPCLearned):
    def __init__(self, use_batch_normalization, activation_fn=nn.ReLU):
        super().__init__(batch_norm=use_batch_normalization, activation=activation_fn)

        self.net = nn.Sequential(
            # in: 7-dim. state of 15 obstacles + 6-dim. drone state = 111
            self._block(in_size=15 * 7 + 6, out_size=256),
            self._block(256, 512),
            self._block(512, 256),
            self._final_block(in_size=256, out_size=3 * 24),
            # out: 3-dim. drone controls x horizon (T-1) of 24 = 72
        )


class MPCLearnedFullSmall(BaseMPCLearned):
    def __init__(self, use_batch_normalization, activation_fn=nn.ReLU):
        super().__init__(batch_norm=use_batch_normalization, activation=activation_fn)

        self.net = nn.Sequential(
            # in: 7-dim. state of 15 obstacles + 6-dim. drone state = 111
            self._block(in_size=15 * 7 + 6, out_size=256),
            self._block(256, 512),
            self._block(512, 256),
            self._final_block(in_size=256, out_size=9 * 11),
            # out: full 9-dim. drone state x horizon (T-1) of 11 = 99
        )
