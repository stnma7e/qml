from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
from tfrecord.torch.dataset import TFRecordDataset


def _merge_stats(
    left: tuple[int, float, float],
    right: tuple[int, float, float],
) -> tuple[int, float, float]:
    left_count, left_mean, left_m2 = left
    right_count, right_mean, right_m2 = right

    if left_count == 0:
        return right
    if right_count == 0:
        return left

    count = left_count + right_count
    delta = right_mean - left_mean
    mean = left_mean + delta * right_count / count
    m2 = left_m2 + right_m2 + delta * delta * left_count * right_count / count
    return count, mean, m2


def _scalar(value: object) -> float:
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    return float(value)


def _stats_for_shard(args: tuple[str, str]) -> tuple[str, int, float, float]:
    shard_path, target = args
    dataset = TFRecordDataset(
        data_path=shard_path,
        index_path=None,
        description={target: "float"},
        shuffle_queue_size=None,
    )

    count = 0
    mean = 0.0
    m2 = 0.0
    for example in dataset:
        count += 1
        value = _scalar(example[target])
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)

    return shard_path, count, mean, m2


def compute_target_stats(
    dataset_dir: Path,
    target: str,
    workers: int,
) -> dict[str, float | int]:
    shards = sorted(dataset_dir.glob("*.tfrecord-*"))
    if not shards:
        raise FileNotFoundError(f"No TFRecord shards found in {dataset_dir}")

    combined = (0, 0.0, 0.0)
    worker_args = [(str(shard), target) for shard in shards]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_stats_for_shard, arg) for arg in worker_args]
        for future in concurrent.futures.as_completed(futures):
            shard_path, count, mean, m2 = future.result()
            combined = _merge_stats(combined, (count, mean, m2))
            print(f"finished {Path(shard_path).name}: {count:,} examples")

    count, mean, m2 = combined
    if count < 2:
        raise ValueError(f"Need at least 2 examples to compute std, got {count}")

    return {
        "count": count,
        "mean": mean,
        "std": (m2 / (count - 1)) ** 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="pbe0_energy")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/qcml/dft_force_field/1.0.0"),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    stats = compute_target_stats(
        dataset_dir=args.dataset_dir,
        target=args.target,
        workers=args.workers,
    )
    print(json.dumps({args.target: stats}, indent=2))


if __name__ == "__main__":
    main()
