
import argparse
import time

import numpy as np
import ray


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
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    ray.init(address="auto")

    print("Cluster resources:", ray.cluster_resources(), flush=True)

    start = time.perf_counter()

    futures = [
        preprocess.remote(
            i,
            args.batch_size,
            args.image_size,
            args.repeats,
        )
        for i in range(args.tasks)
    ]

    results = ray.get(futures)
    elapsed = time.perf_counter() - start

    total_images = sum(result[0] for result in results)
    nodes = {result[1] for result in results}

    print(f"Tasks: {args.tasks}")
    print(f"Images processed: {total_images}")
    print(f"Nodes used: {len(nodes)}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Throughput: {total_images / elapsed:.1f} images/s")


if __name__ == "__main__":
    main()
