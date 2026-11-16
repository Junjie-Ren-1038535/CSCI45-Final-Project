
import csv
import socket
import statistics
import time
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

MATRIX_SIZES = [512, 1024, 2048, 4096]
REPEATS = 5
OPERATIONS = 20

OUTPUT = Path.home() / "optiweave/results/gpu_matmul.csv"


@ray.remote(num_gpus=1)
def benchmark_gpu():
    import torch

    device = torch.device("cuda:0")
    results = []

    for n in MATRIX_SIZES:
        a = torch.randn(n, n, device=device)
        b = torch.randn(n, n, device=device)

        # Warm up the GPU
        for _ in range(10):
            torch.mm(a, b)
        torch.cuda.synchronize()

        timings = []

        for _ in range(REPEATS):
            torch.cuda.synchronize()
            start = time.perf_counter()

            for _ in range(OPERATIONS):
                torch.mm(a, b)

            torch.cuda.synchronize()
            timings.append(
                (time.perf_counter() - start) / OPERATIONS
            )

        median_time = statistics.median(timings)

        # Approximate floating-point operations per matrix multiply
        flops = 2 * (n ** 3)
        gflops = flops / median_time / 1e9

        results.append({
            "hostname": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0),
            "matrix_size": n,
            "median_seconds": round(median_time, 6),
            "gflops": round(gflops, 2),
        })

    return results


def main():
    ray.init(address="auto")

    gpu_nodes = [
        node for node in ray.nodes()
        if node["Alive"]
        and node["Resources"].get("GPU", 0) >= 1
    ]

    print(f"Found {len(gpu_nodes)} GPU nodes")

    tasks = []

    for node in gpu_nodes:
        task = benchmark_gpu.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node["NodeID"],
                soft=False,
            )
        ).remote()
        tasks.append(task)

    results = [
        row
        for node_results in ray.get(tasks)
        for row in node_results
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "hostname",
                "gpu",
                "matrix_size",
                "median_seconds",
                "gflops",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    for row in results:
        print(row)

    print(f"\nResults saved to: {OUTPUT}")
    ray.shutdown()


if __name__ == "__main__":
    main()
