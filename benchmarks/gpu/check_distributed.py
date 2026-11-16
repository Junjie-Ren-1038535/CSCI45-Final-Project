
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

ray.init(address="auto")


@ray.remote(num_gpus=1)
def check_gpu():
    import socket
    import torch

    return {
        "hostname": socket.gethostname(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "distributed": torch.distributed.is_available(),
        "nccl": torch.distributed.is_nccl_available(),
    }


gpu_nodes = [
    node for node in ray.nodes()
    if node["Alive"] and node["Resources"].get("GPU", 0) >= 1
]

tasks = [
    check_gpu.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=node["NodeID"],
            soft=False,
        )
    ).remote()
    for node in gpu_nodes
]

for result in ray.get(tasks):
    print(result)

ray.shutdown()
