
import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, TorchConfig


def train_loop(config):
    import socket

    print("Training worker started", flush=True)

    train.report({
        "hostname": socket.gethostname(),
        "status": "success",
    })


if __name__ == "__main__":
    ray.init(address="auto")

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=True,
        ),
        torch_config=TorchConfig(backend="gloo"),
    )

    result = trainer.fit()
    print("RESULT:", result.metrics)

    ray.shutdown()
