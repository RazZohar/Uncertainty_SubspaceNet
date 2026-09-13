"""Lightweight, file-backed scene subset used by the strict conformal workflow.

Unlike torch.utils.data.Subset(full_dataset, indices), this object stores only the
absolute path to the original dataset.pt and the selected scene indices.  It
therefore avoids duplicating the complete dataset when saving the 80%
development and 20% CP-calibration subsets.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, List

import torch
from torch.utils.data import Dataset


class FileBackedSceneSubset(Dataset):
    def __init__(self, dataset_path: str, indices: Iterable[int], split_name: str = "subset"):
        self.dataset_path = os.path.abspath(dataset_path)
        self.indices: List[int] = [int(i) for i in indices]
        self.split_name = str(split_name)
        self._dataset = None

    def _base_dataset(self):
        if self._dataset is None:
            self._dataset = torch.load(self.dataset_path, weights_only=False)
        return self._dataset

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self._base_dataset()[self.indices[index]]

    def set_use_graph_features(self, use_features: bool = True) -> None:
        dataset = self._base_dataset()
        if hasattr(dataset, "set_use_graph_features"):
            dataset.set_use_graph_features(use_features=use_features)

    def get_sensor_potision(self):  # Keep legacy project spelling for compatibility.
        return self._base_dataset().get_sensor_potision()

    def get_samples_shapes(self):
        dataset = self._base_dataset()
        if hasattr(dataset, "get_samples_shapes"):
            return dataset.get_samples_shapes()
        raise AttributeError("Underlying dataset has no get_samples_shapes method.")

    def __getstate__(self):
        state = self.__dict__.copy()
        # Never serialize the loaded base dataset, which would defeat the point
        # of the file-backed wrapper.
        state["_dataset"] = None
        return state

    def __repr__(self) -> str:
        return (
            f"FileBackedSceneSubset(split_name={self.split_name!r}, "
            f"num_scenes={len(self)}, dataset_path={self.dataset_path!r})"
        )
