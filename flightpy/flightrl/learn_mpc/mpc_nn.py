from torch import nn


class MPCControlLearnedBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            # in: 7-dim. state of 15 obstacles + 6-dim. drone state = 111
            nn.Linear(15 * 7 + 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3 * 24),
            # out: 3-dim. drone controls x horizon (T-1) of 24 = 72
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu(x)
        return logits


class MPCControlLearned(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            # in: 7-dim. state of 15 obstacles + 6-dim. drone state = 111
            nn.Linear(15 * 7 + 6, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 24),
            # out: 3-dim. drone controls x horizon (T-1) of 24 = 72
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu(x)
        return logits


class MPCControlLearnedSmallBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            # in: 7-dim. state of 15 obstacles + 6-dim. drone state = 111
            nn.Linear(15 * 7 + 6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3 * 24),
            # out: 3-dim. drone controls x horizon (T-1) of 24 = 72
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu(x)
        return logits


class MPCControllLearnedSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            # in: 7-dim. state of 15 obstacles + 6-dim. drone state = 111
            nn.Linear(15 * 7 + 6, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 24),
            # out: 3-dim. drone controls x horizon (T-1) of 24 = 72
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu(x)
        return logits


class MPCFullLearned(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            # in: 7-dim. state of 15 obstacles + 6-dim. drone state = 111
            nn.Linear(15 * 7 + 6, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9 * 11),
            # out: full 9-dim. drone state x horizon (T-1) of 11 = 99
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu(x)
        return logits
