import os

import numpy as np
from torch.utils.data import Dataset


class NMPCDataset(Dataset):
    """Nonlinear MPC dataset"""

    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = self._get_data_dir(root_dir)
        self.metadata = self._read_metadata()

        self.buckets = np.cumsum([v[0] for _, v in self.metadata.items()])
        self.dataset = np.load(self.root_dir + "dataset_hard.npz")

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return sum([v[0] for _, v in self.metadata.items()])

    def __getitem__(self, idx):
        bucket = self._get_bucket(idx, start_index=0, end_index=len(self.buckets) - 1)
        observation = self.dataset[f"input_hard_{bucket}"][
            idx % self.buckets[bucket - 1] if bucket != 0 else idx, :
        ]
        controls = self.dataset[f"output_hard_{bucket}"][
            idx % self.buckets[bucket - 1] if bucket != 0 else idx, :
        ]

        if self.transform:
            observation = self.transform(observation)
        if self.target_transform:
            controls = self.target_transform(controls)

        return observation, controls

    def _get_bucket(self, i, start_index, end_index):
        """Binary search for the corresponding bucket"""
        if start_index + 1 == end_index:
            if i < self.buckets[start_index]:
                return start_index
            else:
                return end_index
        middle = (start_index + end_index) // 2
        if i >= self.buckets[middle]:
            return self._get_bucket(i, start_index=middle, end_index=end_index)
        else:
            return self._get_bucket(i, start_index=start_index, end_index=middle)

    @staticmethod
    def _get_data_dir(root):
        pkg = "agile_flight"
        cwd = os.getcwd()
        return (
            cwd[: cwd.find(pkg) + len(pkg) + 1]
            + "flightmare/flightpy/datasets/"
            + root
            + "/"
        )

    def _read_metadata(self):
        metadata = dict()
        with open(self.root_dir + "metadata.txt", "r+") as metadata_file:
            metadata_lines = metadata_file.read().splitlines()
            for line in metadata_lines:
                env = line.split()[2].rstrip(":")
                collisions = int(line.split()[-2])
                size = int(line.split()[4])
                metadata[env] = (size, collisions)
        return metadata
