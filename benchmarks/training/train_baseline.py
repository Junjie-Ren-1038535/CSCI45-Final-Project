
import argparse
import csv
import socket
import time
from pathlib import Path
from contextlib import nullcontext

import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer


def train_func(config):
    import os

    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET"
    os.environ["NCCL_DEBUG_FILE"] = "/tmp/nccl-debug.%h.%p.log"
    os.environ["NCCL_SOCKET_IFNAME"] = "enp4s0,enp5s0"

    import torch
    from torch import nn
    from ray import train
    from ray.train.torch import prepare_model, get_device

    torch.manual_seed(42)

    # Synthetic classification dataset generated locally
    batch_size = config["batch_size"]
    steps = config["steps"]

    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10),
    )

    model = prepare_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    use_amp = False
    torch.backends.cuda.matmul.allow_tf32 = True

    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    device = get_device()

    start = time.perf_counter()

    accumulation_steps = 128

    optimizer.zero_grad()

    for step in range(steps):
        x = torch.randn(batch_size, 1024, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)

        should_sync = (
           (step + 1) % accumulation_steps == 0
           or (step + 1) == steps
        )


        sync_context = (
            model.no_sync()
            if not should_sync and hasattr(model, "no_sync")
            else nullcontext()
        )

        with sync_context:
           with torch.autocast(
              device_type="cuda",
              dtype=torch.float16,
              enabled=use_amp
           ):
              output = model(x)
              loss = loss_fn(output, y)

           remaining = steps % accumulation_steps

           current_accumulation = (
                  remaining
                  if remaining and step >= steps - remaining
                  else accumulation_steps
           )

           loss = loss / current_accumulation
           scaler.scale(loss).backward()

        if should_sync:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    context = train.get_context()
    world_size = context.get_world_size()

    train.report({
        "hostname": socket.gethostname(),
        "elapsed_seconds": elapsed,
        "samples_per_second": (
            batch_size * steps * world_size / elapsed
        ),
        "world_size": world_size,
        "batch_size": batch_size,
        "steps": steps,
        "loss": loss.item() * current_accumulation,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
	"--gpu",
	choices=["auto", "rtx3060", "gtx1050"],
	default="auto",
	)
    args = parser.parse_args()

    ray.init(address="auto")

    gpu_resources = {
	"rtx3060": {"node:10.10.10.20": 0.001},
	"gtx1050": {"node:100.100.235.122": 0.001},

    }

    resources_per_worker = {"GPU": 1}

    if args.gpu != "auto":
        if args.workers != 1:
            parser.error("--gpu selection requires --workers 1")

        resources_per_worker.update(gpu_resources[args.gpu])

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "batch_size": args.batch_size,
            "steps": args.steps,
        },
        scaling_config=ScalingConfig(
            num_workers=args.workers,
            use_gpu=True,
	    resources_per_worker=resources_per_worker,
        ),
    )

    result = trainer.fit()
    print(result.metrics)

    ray.shutdown()


if __name__ == "__main__":
    main()
