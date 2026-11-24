
import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torchvision.models import resnet18, resnet50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Maximum batches per epoch; 0 means all batches",
    )
    parser.add_argument(
    "--checkpointing",
    action="store_true",
    help="Enable activation checkpointing",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Input image resolution (32, 64, 128, or 224)",
    )
    parser.add_argument(
        "--data-dir",
        default="/home/a/homelab-data/datasets",
    )
    parser.add_argument(
        "--model",
        choices=["resnet18", "resnet50"],
        default="resnet18",
    )
    args = parser.parse_args()

    

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ])

    train_data = CIFAR100(
        root=args.data_dir,
        train=True,
        download=False,
        transform=transform,
    )

    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = (
        resnet50(num_classes=100)
        if args.model == "resnet50"
        else resnet18(num_classes=100)
        ).to(device)

    if args.checkpointing:
        from torch.utils.checkpoint import checkpoint

        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(model, layer_name)
            original_forward = layer.forward

            def checkpointed_forward(x, forward_fn=original_forward):
                return checkpoint(
                    forward_fn,
                    x,
                    use_reentrant=False,
                )

            layer.forward = checkpointed_forward

        print("Activation checkpointing: ENABLED")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        samples = 0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()

        for batch_idx, (images, labels) in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            samples += images.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        peak_vram = (
            torch.cuda.max_memory_allocated() / 1024**3
            if device.type == "cuda" else 0
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {total_loss / samples:.4f} | "
            f"Time: {elapsed:.2f}s | "
            f"Throughput: {samples / elapsed:.1f} samples/s | "
            f"Peak VRAM: {peak_vram:.2f} GiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
