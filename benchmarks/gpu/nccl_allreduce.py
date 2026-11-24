import argparse
import csv
import socket
import time
from pathlib import Path

import ray
import torch
import torch.distributed as dist

from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, get_device


def benchmark_allreduce(config):
    from ray import train

    device = get_device()
    rank = train.get_context().get_world_rank()

    sizes_mb = [0.004, 0.016, 0.064, 0.25, 1, 4, 16, 32]
    iterations = 20
    warmup = 5

    results = []

    for size_mb in sizes_mb:
        num_elements = int(size_mb * 1024 * 1024 // 4)

        tensor = torch.ones(
            num_elements,
            dtype=torch.float32,
            device=device,
        )

        # Warm up NCCL before measuring.
        for _ in range(warmup):
            dist.all_reduce(tensor)

        torch.cuda.synchronize()
        dist.barrier()

        start = time.perf_counter()

        for _ in range(iterations):
            dist.all_reduce(tensor)

        torch.cuda.synchronize()
        dist.barrier()

        elapsed = time.perf_counter() - start

        if rank == 0:
            latency_ms = elapsed / iterations * 1000
            bandwidth_mbps = (
                size_mb * iterations / elapsed
            )

            results.append({
                "size_mb": size_mb,
                "latency_ms": latency_ms,
                "bandwidth_MBps": bandwidth_mbps,
            })

            print(
                f"Size: {size_mb:>2} MB | "
                f"Latency: {latency_ms:.3f} ms | "
                f"Bandwidth: {bandwidth_mbps:.2f} MB/s",
                flush=True,
            )

    train.report({"results": results})


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark NCCL AllReduce across GPUs"
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if args.workers < 2:
        parser.error("NCCL communication benchmark requires at least 2 workers")

    ray.init(address="auto")

    trainer = TorchTrainer(
        train_loop_per_worker=benchmark_allreduce,
        train_loop_config={},
        scaling_config=ScalingConfig(
            num_workers=args.workers,
            use_gpu=True,
        ),
    )

    result = trainer.fit()
    print("NCCL benchmark finished.")
    print("Result metrics:", result.metrics)

    ray.shutdown()


if __name__ == "__main__":
    main()
