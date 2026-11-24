
import argparse
import csv
import socket
import time
from pathlib import Path
from contextlib import nullcontext

import ray
from ray.train import ScalingConfig, RunConfig
from benchmarks.training.metrics_callback import BenchmarkMetricsCallback
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

    if config.get("compression", "none") == "fp16":
        from torch.distributed.algorithms.ddp_comm_hooks import (
            default_hooks,
        )

        if not isinstance(
            model, torch.nn.parallel.DistributedDataParallel
        ):
            raise RuntimeError(
                "FP16 compression requires DistributedDataParallel"
            )

        model.register_comm_hook(
            state=None,
            hook=default_hooks.fp16_compress_hook,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    use_amp = False
    torch.backends.cuda.matmul.allow_tf32 = True

    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    device = get_device()

    # Synchronize all workers before starting the benchmark
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    torch.cuda.synchronize()
    start = time.perf_counter()

    accumulation_steps = config["accumulation"]

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

    # Use the slowest worker's elapsed time for distributed throughput.
    if torch.distributed.is_initialized():
   	 elapsed_tensor = torch.tensor(
        	[elapsed],
        	dtype=torch.float64,
        	device=device,
    	)
   	 torch.distributed.all_reduce(
       		 elapsed_tensor,
       		 op=torch.distributed.ReduceOp.MAX,
   	 )
   	 global_elapsed = elapsed_tensor.item()
    else:
   	 global_elapsed = elapsed

    context = train.get_context()
    world_size = context.get_world_size()

    metrics = {
        "hostname": socket.gethostname(),
        "elapsed_seconds": elapsed,
        "samples_per_second": (
            batch_size * steps * world_size / global_elapsed
        ),
        "global_elapsed_seconds": global_elapsed,
        "accumulation": accumulation_steps,
        "world_size": world_size,
        "batch_size": batch_size,
        "steps": steps,
        "loss": loss.item() * current_accumulation,
    }

    train.report(metrics)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
   	"--accumulation",
	type=int,
	default=128,
	help="Number of microbatches per optimizer update",
    )
    parser.add_argument(
	"--gpu",
	choices=["auto", "rtx3060", "gtx1050"],
	default="auto",
    )
    parser.add_argument(
    "--compression",
    choices=["none", "fp16"],
    default="none",
    help="DDP gradient communication compression",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/training_benchmarks.csv",
        help="CSV file for recording training benchmark results",
    )
    args = parser.parse_args()
    if args.workers < 1:
   	 parser.error("--workers must be at least 1")

    if args.batch_size < 1:
   	 parser.error("--batch-size must be at least 1")

    if args.steps < 1:
   	 parser.error("--steps must be at least 1")

    if args.accumulation < 1:
    	parser.error("--accumulation must be at least 1")

    ray.init(
        address="auto",
        runtime_env={
            "working_dir": "/home/a/optiweave-github"
        },
    )

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
	        "accumulation": args.accumulation,
            "compression": args.compression,
        },
        run_config=RunConfig(
            callbacks=[
                BenchmarkMetricsCallback(
                    str(Path(args.output).resolve())
                )
            ],
        ),
        scaling_config=ScalingConfig(
            num_workers=args.workers,
            use_gpu=True,
	    resources_per_worker=resources_per_worker,
        ),
    )

    result = trainer.fit()

    print("Result type:", type(result))
    print("Result metrics:", result.metrics)
    print("Result metrics dataframe:", result.metrics_dataframe)
    print("Result checkpoint:", result.checkpoint)

    ray.shutdown()


if __name__ == "__main__":
    main()
