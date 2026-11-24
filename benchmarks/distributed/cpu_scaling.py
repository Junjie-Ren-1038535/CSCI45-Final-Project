import argparse
import time
import ray
import numpy as np
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=1)
def preprocess(batch_id, batch_size, image_size, repeats):
    rng = np.random.default_rng(batch_id)

    images = rng.integers(
        0, 256,
        size=(batch_size, image_size, image_size, 3),
        dtype=np.uint8,
    )

    for _ in range(repeats):
        x = images.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.25
        x = np.clip(x, -2.0, 2.0)
        images = ((x + 2.0) * 63.75).astype(np.uint8)

    return batch_size, ray.get_runtime_context().get_node_id()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    ray.init(address="auto")

    # Select active CPU-only worker nodes.
    workers = [
        node for node in ray.nodes()
        if node["Alive"]
        and node["Resources"].get("CPU", 0) > 0
    ]

    workers.sort(key=lambda n: n["NodeManagerAddress"])

    selected = workers[:args.nodes]

    if len(selected) < args.nodes:
        raise RuntimeError(
            f"Requested {args.nodes} nodes, found {len(selected)} CPU workers"
        )

    print("Selected workers:")
    for node in selected:
        print(" ", node["NodeManagerAddress"])

    start = time.perf_counter()

    
    # Allocate tasks proportionally to each node's CPU resources.
    weights = [
        int(node["Resources"]["CPU"])
        for node in selected
    ]
    total_weight = sum(weights)

    task_counts = [
        args.tasks * weight // total_weight
        for weight in weights
    ]

    remaining = args.tasks - sum(task_counts)
    order = sorted(
        range(len(selected)),
        key=lambda i: (args.tasks * weights[i]) % total_weight,
        reverse=True,
    )

    for index in order[:remaining]:
        task_counts[index] += 1

    assignments = [
        node
        for node, count in zip(selected, task_counts)
        for _ in range(count)
    ]

    print("Task allocation:")
    for node, count in zip(selected, task_counts):
        print(
            f"  {node['NodeManagerAddress']}: "
            f"{count} tasks / "
            f"{node['Resources']['CPU']:g} CPUs"
        )

    start = time.perf_counter()
    futures = []

    for i, node in enumerate(assignments):
        futures.append(
            preprocess.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"],
                    soft=False,
                )
            ).remote(
                i,
                args.batch_size,
                args.image_size,
                args.repeats,
            )
        )

    results = ray.get(futures)
    elapsed = time.perf_counter() - start

    images = sum(r[0] for r in results)
    used_nodes = len(set(r[1] for r in results))

    print(f"Requested nodes: {args.nodes}")
    print(f"Actual nodes used: {used_nodes}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Throughput: {images / elapsed:.1f} images/s")


if __name__ == "__main__":
    main()
