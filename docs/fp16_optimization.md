# FP16 Gradient Communication Optimization

## Objective

Evaluate whether FP16 gradient compression improves distributed
training performance across heterogeneous GPUs connected by
a 1 Gbps Ethernet network.

## Hardware

- NVIDIA RTX 3060 (12 GB)
- NVIDIA GTX 1050 (4 GB)
- Ray Train with PyTorch DistributedDataParallel (DDP)
- NCCL communication backend

## Methodology

The experiment compares standard FP32 gradient communication
against PyTorch's fp16_compress_hook.

Configuration:
- 2 GPU workers
- Batch size: 64 per worker
- 128 training steps per worker
- Gradient accumulation: 128
- Synthetic classification workload
- Two measurements per configuration

## Results

| Configuration | Average throughput (samples/s) |
|---|---:|
| FP32 communication | 32,569 |
| FP16 compression | 37,429 |

FP16 compression increased measured throughput by approximately
14.9% compared with standard FP32 communication.

Average measured training time decreased from approximately
0.5033 seconds to 0.4377 seconds.

## Interpretation

FP16 compression reduces the communicated gradient payload
from approximately 16.05 MiB to 8.03 MiB.

This reduces communication overhead over the network and
improves throughput in the tested configuration.

## Limitations

- Measurements cover a very short synthetic training workload.
- Only two repetitions were performed per configuration.
- No dedicated warmup phase was included in the timed benchmark.
- Model convergence and final accuracy were not evaluated.
- Results may differ with larger models or different networks.

## Data

See `results/fp16_comparison.csv` for individual measurements.
