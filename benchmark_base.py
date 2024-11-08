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
        self.batch_sizes = [1, 8, 16, 32]  # Removed larger batch sizes

        # Reduced matrix sizes
        self.matrix_sizes = [128, 256, 512]  # Removed larger matrix sizes

        # Reduced number of iterations
        self.warmup_iterations = 5  # Reduced from 10
        self.test_iterations = 50  # Reduced from 100

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

    def benchmark_batch_norm(self, batch_size: int):
        """Benchmark batch normalization"""
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

    def save_results(self, config: BenchmarkConfig) -> None:
        """Save benchmark results to JSON"""
        results_dict = {
            "config": config.__dict__,
            "results": [r.to_dict() for r in self.results],
        }

        output_path = Path("benchmark_results.json")
        with output_path.open("w") as f:
            json.dump(results_dict, f, indent=2)

        logging.info("Saved benchmark results to %s", output_path)

    def plot_results(self) -> None:
        """Generate plots for benchmark results"""
        operations = set(r.operation for r in self.results)

        for op in operations:
            op_results = [r for r in self.results if r.operation == op]

            # Group by parameter values
            param_key = list(op_results[0].parameters.keys())[0]
            param_values = sorted(set(r.parameters[param_key] for r in op_results))

            pytorch_times = []
            jax_times = []

            for param_val in param_values:
                pytorch_result = next(
                    r
                    for r in op_results
                    if r.framework == "pytorch" and r.parameters[param_key] == param_val
                )
                jax_result = next(
                    r
                    for r in op_results
                    if r.framework == "jax" and r.parameters[param_key] == param_val
                )

                pytorch_times.append(pytorch_result.mean_time * 1000)
                jax_times.append(jax_result.mean_time * 1000)

            plt.figure(figsize=(10, 6))
            plt.plot(param_values, pytorch_times, "o-", label="PyTorch")
            plt.plot(param_values, jax_times, "o-", label="JAX")
            plt.xlabel(param_key)
            plt.ylabel("Time (ms)")
            plt.title(f"{op} Performance Comparison")
            plt.legend()
            plt.grid(True)

            output_path = f"benchmark_{op}.png"
            plt.savefig(output_path)
            plt.close()

            logging.info("Saved %s benchmark plot to %s", op, output_path)
