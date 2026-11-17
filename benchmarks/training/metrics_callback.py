
import csv
from pathlib import Path

from ray.train import UserCallback


class BenchmarkMetricsCallback(UserCallback):
    """Save Ray Train worker metrics to a CSV file."""

    def __init__(self, output_path):
        self.output_path = Path(output_path)

    def after_report(self, run_context, metrics, checkpoint):
        print(
            f"[OptiWeave Callback] after_report invoked;"
            f"metrics={metrics}",
            flush=True,
        )
        if not metrics:
            return

        # Ray Train reports one metrics dictionary per worker.
        # All workers share the same cluster-wide throughput.
        valid_metrics = [m for m in metrics if m]

        if not valid_metrics:
            return

        record = valid_metrics[0].copy()
        record["hostname"] = "+".join(
            m["hostname"] for m in valid_metrics
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = self.output_path.exists()
        fieldnames = list(record.keys())

        with self.output_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if not file_exists or self.output_path.stat().st_size == 0:
                writer.writeheader()

            writer.writerow(record)

        print(f"Benchmark saved to {self.output_path}")
