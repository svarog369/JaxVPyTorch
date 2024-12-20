from dataclasses import dataclass
from typing import Any, Dict, List
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt


@dataclass
class BenchmarkResult:
    """Store results of a benchmark run"""

    name: str
    framework: str
    operation: str
    mean_time: float
    std_time: float
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "framework": self.framework,
            "operation": self.operation,
            "mean_time_ms": self.mean_time * 1000,
            "std_time_ms": self.std_time * 1000,
            "parameters": self.parameters,
        }


@dataclass
class NNConfig:
    """Neural network specific configuration"""

    hidden_sizes: List[int] | None = None
    sequence_lengths: List[int] | None = None
    embedding_dims: List[int] | None = None
    num_heads: List[int] | None = None

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [64, 128, 256, 512]
        if self.sequence_lengths is None:
            self.sequence_lengths = [16, 32, 64, 128]
        if self.embedding_dims is None:
            self.embedding_dims = [32, 64, 128, 256]
        if self.num_heads is None:
            self.num_heads = [1, 2, 4, 8]


class BenchmarkConfig:
    """Configuration for benchmark runs with memory-efficient defaults"""

    def __init__(self):
        # Reduced batch sizes
        self.batch_sizes = [32, 64, 128, 256, 512, 600]

        # Reduced matrix sizes
        self.matrix_sizes = [128, 256, 512, 1024]

        # Reduced number of iterations
        self.warmup_iterations = 10
        self.test_iterations = 100

        self.seed = 42
        self.nn_config = NNConfig()


class BenchmarkBase:
    """Base class for framework-specific benchmarks"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []
        self._initialize()

    def _initialize(self):
        """Framework-specific initialization"""
        raise NotImplementedError

    def benchmark_matmul(self, size: int):
        """Benchmark matrix multiplication"""
        raise NotImplementedError

    def benchmark_convolution(self, batch_size: int):
        """Benchmark 2D convolution"""
        raise NotImplementedError

    def benchmark_gradient(self, size: int):
        """Benchmark gradient computation"""
        raise NotImplementedError

    def benchmark_transformer_layer(
        self, batch_size: int, seq_len: int, hidden_size: int, num_heads: int
    ):
        """Benchmark transformer layer forward pass"""
        raise NotImplementedError

    def benchmark_lstm(self, batch_size: int, seq_len: int, hidden_size: int):
        """Benchmark LSTM forward and backward pass"""
        raise NotImplementedError

    def benchmark_mlp(self, batch_size: int, input_size: int, hidden_sizes: List[int]):
        """Benchmark MLP forward and backward pass"""
        raise NotImplementedError

    def benchmark_attention(self, batch_size: int, seq_len: int, hidden_size: int):
        """Benchmark attention mechanism"""
        raise NotImplementedError

    def benchmark_embedding(self, batch_size: int, vocab_size: int, embedding_dim: int):
        """Benchmark embedding layer"""
        raise NotImplementedError

    def benchmark_optimizer_step(self, num_params: int, optimizer: str):
        """Benchmark optimizer update step"""
        raise NotImplementedError


class ResultManager:
    """Manages benchmark results and visualization"""

    def __init__(self):
        self.results = []

    def add_results(self, results):
        self.results.extend(results)

    def _config_to_dict(self, config):
        """Convert config object to a JSON-serializable dictionary"""
        config_dict = {
            "batch_sizes": config.batch_sizes,
            "matrix_sizes": config.matrix_sizes,
            "warmup_iterations": config.warmup_iterations,
            "test_iterations": config.test_iterations,
            "seed": config.seed,
            "nn_config": {
                "hidden_sizes": config.nn_config.hidden_sizes,
                "sequence_lengths": config.nn_config.sequence_lengths,
                "embedding_dims": config.nn_config.embedding_dims,
                "num_heads": config.nn_config.num_heads,
            },
        }
        return config_dict

    def save_results(self, config: BenchmarkConfig) -> None:
        """Save benchmark results to JSON"""
        results_dict = {
            "config": self._config_to_dict(config),
            "results": [r.to_dict() for r in self.results],
        }

        output_path = Path("benchmark_results.json")
        with output_path.open("w") as f:
            json.dump(results_dict, f, indent=2)

        logging.info("Saved benchmark results to %s", output_path)

    def plot_results(self) -> None:
        """Generate detailed plots for benchmark results with min, max, and mean statistics"""
        operations = set(r.operation for r in self.results)

        for op in operations:
            op_results = [r for r in self.results if r.operation == op]

            # Group by framework
            pytorch_results = [r for r in op_results if r.framework == "pytorch"]
            jax_results = [r for r in op_results if r.framework == "jax"]

            if not pytorch_results or not jax_results:
                continue

            # Identify the parameter to plot
            param_keys = list(op_results[0].parameters.keys())
            main_param_key = next(
                (
                    k
                    for k in ["size", "batch_size", "hidden_size", "num_params"]
                    if k in param_keys
                ),
                param_keys[0],
            )

            # Prepare data for plotting
            param_values = sorted(set(r.parameters[main_param_key] for r in op_results))

            pytorch_times = []
            jax_times = []
            pytorch_stds = []
            jax_stds = []

            for param_val in param_values:
                # Get results for this parameter value
                pytorch_result = next(
                    (
                        r
                        for r in pytorch_results
                        if r.parameters[main_param_key] == param_val
                    ),
                    None,
                )
                jax_result = next(
                    (
                        r
                        for r in jax_results
                        if r.parameters[main_param_key] == param_val
                    ),
                    None,
                )

                if pytorch_result and jax_result:
                    pytorch_times.append(pytorch_result.mean_time * 1000)
                    jax_times.append(jax_result.mean_time * 1000)
                    pytorch_stds.append(pytorch_result.std_time * 1000)
                    jax_stds.append(jax_result.std_time * 1000)

            if pytorch_times and jax_times:
                plt.figure(figsize=(12, 8))

                # Plot with error bars
                plt.errorbar(
                    param_values,
                    pytorch_times,
                    yerr=pytorch_stds,
                    fmt="o-",
                    label="PyTorch",
                    capsize=5,
                    color="#EE4C2C",
                )
                plt.errorbar(
                    param_values,
                    jax_times,
                    yerr=jax_stds,
                    fmt="o-",
                    label="JAX",
                    capsize=5,
                    color="#00A67E",
                )

                # Add min/max annotations
                pytorch_min = min(pytorch_times)
                pytorch_max = max(pytorch_times)
                pytorch_mean = sum(pytorch_times) / len(pytorch_times)

                jax_min = min(jax_times)
                jax_max = max(jax_times)
                jax_mean = sum(jax_times) / len(jax_times)

                # Add statistics box
                stats_text = (
                    f"PyTorch Stats (ms):\n"
                    f"Min: {pytorch_min:.3f}\n"
                    f"Max: {pytorch_max:.3f}\n"
                    f"Mean: {pytorch_mean:.3f}\n\n"
                    f"JAX Stats (ms):\n"
                    f"Min: {jax_min:.3f}\n"
                    f"Max: {jax_max:.3f}\n"
                    f"Mean: {jax_mean:.3f}"
                )

                plt.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                )

                plt.xlabel(main_param_key.replace("_", " ").title())
                plt.ylabel("Time (ms)")
                plt.title(f'{op.replace("_", " ").title()} Performance Comparison')
                plt.legend()
                plt.grid(True, which="both", ls="-", alpha=0.2)

                # Use log scale if max/min ratio is large
                if (
                    max(max(pytorch_times), max(jax_times))
                    / min(min(pytorch_times), min(jax_times))
                    > 10
                ):
                    plt.yscale("log")
                    plt.grid(True, which="minor", ls=":", alpha=0.1)

                # Ensure some padding around the data
                plt.margins(x=0.1)

                output_path = f"benchmark_{op}.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close()

                logging.info("Saved %s benchmark plot to %s", op, output_path)
                logging.info("%s benchmark statistics:\n%s", op, stats_text)
