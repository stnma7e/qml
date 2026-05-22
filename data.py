from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import DataLoader, IterableDataset

DATASET_DIR = Path("data/qcml/dft_force_field/1.0.0")
METADATA_DIR = Path("data/qcml/dft_metadata/1.0.0")
TFRECORD_GLOB = "*.tfrecord-*"

# Matches the TFRecord schema for `qcml/dft_force_field`.
TFRECORD_DESCRIPTION = {
    "pbe0_formation_energy": "float",
    "pbe0_forces": "float",
    "key_hash": "byte",
    "multiplicity": "int",
    "charge": "int",
    "is_outlier": "int",
    "atomic_numbers": "int",
    "positions": "float",
    "pbe0_energy": "float",
}

METADATA_TFRECORD_DESCRIPTION = {
    "molecular_weight": "float",
    "key_hash": "byte",
    "chemical_formula": "byte",
    "multiplicity": "int",
    "conformation_seq": "int",
    "charge": "int",
    "conformation_parent_seq": "int",
    "smiles_hash": "byte",
    "smiles": "byte",
    "num_heavy_atoms": "int",
    "num_atoms": "int",
}


class DFTForceFieldDataset(IterableDataset[dict[str, Any]]):
    def __init__(self, dataset_dir: Path | str = DATASET_DIR) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.shards = sorted(self.dataset_dir.glob(TFRECORD_GLOB))
        if not self.shards:
            raise FileNotFoundError(f"No TFRecord shards found in: {self.dataset_dir}")

    @staticmethod
    def _as_scalar(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            if value.size == 1:
                return value.reshape(()).item()
        return value

    @staticmethod
    def _to_sample(example: dict[str, Any]) -> dict[str, Any]:
        key_hash = example["key_hash"]
        if isinstance(key_hash, bytes):
            key_hash = key_hash.decode("utf-8")

        positions = torch.as_tensor(example["positions"], dtype=torch.float32)
        if positions.ndim == 1 and positions.numel() % 3 == 0:
            positions = positions.view(-1, 3)

        pbe0_forces = torch.as_tensor(example["pbe0_forces"], dtype=torch.float32)
        if pbe0_forces.ndim == 1 and pbe0_forces.numel() % 3 == 0:
            pbe0_forces = pbe0_forces.view(-1, 3)

        return {
            "key_hash": key_hash,
            "positions": positions,
            "pbe0_forces": pbe0_forces,
            "atomic_numbers": torch.as_tensor(
                example["atomic_numbers"], dtype=torch.int64
            ),
            "charge": int(DFTForceFieldDataset._as_scalar(example["charge"])),
            "multiplicity": int(
                DFTForceFieldDataset._as_scalar(example["multiplicity"])
            ),
            "is_outlier": bool(DFTForceFieldDataset._as_scalar(example["is_outlier"])),
            "pbe0_energy": float(
                DFTForceFieldDataset._as_scalar(example["pbe0_energy"])
            ),
            "pbe0_formation_energy": float(
                DFTForceFieldDataset._as_scalar(example["pbe0_formation_energy"])
            ),
        }

    def __iter__(self):
        for shard in self.shards:
            shard_ds = TFRecordDataset(
                data_path=str(shard),
                index_path=None,
                description=TFRECORD_DESCRIPTION,
                transform=self._to_sample,
                shuffle_queue_size=None,
            )
            yield from shard_ds


class DFTMetadataDataset(IterableDataset[dict[str, Any]]):
    def __init__(self, dataset_dir: Path | str = METADATA_DIR) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.shards = sorted(self.dataset_dir.glob(TFRECORD_GLOB))
        if not self.shards:
            raise FileNotFoundError(f"No TFRecord shards found in: {self.dataset_dir}")

    @staticmethod
    def _decode_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _to_sample(example: dict[str, Any]) -> dict[str, Any]:
        return {
            "key_hash": DFTMetadataDataset._decode_text(example["key_hash"]),
            "chemical_formula": DFTMetadataDataset._decode_text(
                example["chemical_formula"]
            ),
            "smiles_hash": DFTMetadataDataset._decode_text(example["smiles_hash"]),
            "smiles": DFTMetadataDataset._decode_text(example["smiles"]),
            "molecular_weight": float(
                DFTForceFieldDataset._as_scalar(example["molecular_weight"])
            ),
            "multiplicity": int(DFTForceFieldDataset._as_scalar(example["multiplicity"])),
            "conformation_seq": int(
                DFTForceFieldDataset._as_scalar(example["conformation_seq"])
            ),
            "charge": int(DFTForceFieldDataset._as_scalar(example["charge"])),
            "conformation_parent_seq": int(
                DFTForceFieldDataset._as_scalar(example["conformation_parent_seq"])
            ),
            "num_heavy_atoms": int(
                DFTForceFieldDataset._as_scalar(example["num_heavy_atoms"])
            ),
            "num_atoms": int(DFTForceFieldDataset._as_scalar(example["num_atoms"])),
        }

    def __iter__(self):
        for shard in self.shards:
            shard_ds = TFRecordDataset(
                data_path=str(shard),
                index_path=None,
                description=METADATA_TFRECORD_DESCRIPTION,
                transform=self._to_sample,
                shuffle_queue_size=None,
            )
            yield from shard_ds


def collate_force_field(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "key_hash": [item["key_hash"] for item in batch],
        "positions": [item["positions"] for item in batch],
        "pbe0_forces": [item["pbe0_forces"] for item in batch],
        "atomic_numbers": [item["atomic_numbers"] for item in batch],
        "charge": torch.tensor([item["charge"] for item in batch], dtype=torch.int64),
        "multiplicity": torch.tensor(
            [item["multiplicity"] for item in batch], dtype=torch.int64
        ),
        "is_outlier": torch.tensor(
            [item["is_outlier"] for item in batch], dtype=torch.bool
        ),
        "pbe0_energy": torch.tensor(
            [item["pbe0_energy"] for item in batch], dtype=torch.float32
        ),
        "pbe0_formation_energy": torch.tensor(
            [item["pbe0_formation_energy"] for item in batch], dtype=torch.float32
        ),
    }


def collate_metadata(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "key_hash": [item["key_hash"] for item in batch],
        "chemical_formula": [item["chemical_formula"] for item in batch],
        "smiles_hash": [item["smiles_hash"] for item in batch],
        "smiles": [item["smiles"] for item in batch],
        "molecular_weight": torch.tensor(
            [item["molecular_weight"] for item in batch], dtype=torch.float32
        ),
        "multiplicity": torch.tensor(
            [item["multiplicity"] for item in batch], dtype=torch.int64
        ),
        "conformation_seq": torch.tensor(
            [item["conformation_seq"] for item in batch], dtype=torch.int64
        ),
        "charge": torch.tensor([item["charge"] for item in batch], dtype=torch.int64),
        "conformation_parent_seq": torch.tensor(
            [item["conformation_parent_seq"] for item in batch], dtype=torch.int64
        ),
        "num_heavy_atoms": torch.tensor(
            [item["num_heavy_atoms"] for item in batch], dtype=torch.int64
        ),
        "num_atoms": torch.tensor([item["num_atoms"] for item in batch], dtype=torch.int64),
    }


class AtomCountBatchedDataset(IterableDataset[dict[str, Any]]):
    """Groups samples by atom count and emits homogeneous pre-batched dicts."""

    def __init__(
        self,
        base_dataset: IterableDataset[dict[str, Any]],
        batch_size: int,
        drop_last: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        buffers: dict[int, list[dict[str, Any]]] = {}

        for sample in self.base_dataset:
            atom_count = int(sample["positions"].shape[0])
            bucket = buffers.setdefault(atom_count, [])
            bucket.append(sample)

            if len(bucket) == self.batch_size:
                yield collate_force_field(bucket)
                buffers[atom_count] = []

        if not self.drop_last:
            for bucket in buffers.values():
                if bucket:
                    yield collate_force_field(bucket)


class KeyHashZippedDataset(IterableDataset[dict[str, Any]]):
    """Zips force-field and metadata streams and validates key alignment."""

    def __init__(
        self,
        force_field_dataset: IterableDataset[dict[str, Any]],
        metadata_dataset: IterableDataset[dict[str, Any]],
    ) -> None:
        self.force_field_dataset = force_field_dataset
        self.metadata_dataset = metadata_dataset

    def __iter__(self):
        for ff, md in zip(self.force_field_dataset, self.metadata_dataset):
            if ff["key_hash"] != md["key_hash"]:
                raise ValueError(
                    "Key mismatch between force_field and metadata streams: "
                    f"{ff['key_hash']} != {md['key_hash']}"
                )
            yield {**ff, **md}


def load_data(
    dataset_dir: Path | str = DATASET_DIR,
    batch_size: int = 64,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    dataset = DFTForceFieldDataset(dataset_dir=dataset_dir)
    batched_dataset = AtomCountBatchedDataset(
        base_dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
    )
    return DataLoader(
        batched_dataset,
        batch_size=None,
        num_workers=num_workers,
    )


def load_metadata_data(
    dataset_dir: Path | str = METADATA_DIR,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:
    dataset = DFTMetadataDataset(dataset_dir=dataset_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_metadata,
    )


def load_joined_data(
    force_field_dir: Path | str = DATASET_DIR,
    metadata_dir: Path | str = METADATA_DIR,
    batch_size: int = 64,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    force_field_dataset = DFTForceFieldDataset(force_field_dir)
    metadata_dataset = DFTMetadataDataset(metadata_dir)
    zipped = KeyHashZippedDataset(force_field_dataset, metadata_dataset)
    batched_dataset = AtomCountBatchedDataset(
        base_dataset=zipped,
        batch_size=batch_size,
        drop_last=drop_last,
    )
    return DataLoader(
        batched_dataset,
        batch_size=None,
        num_workers=num_workers,
    )
