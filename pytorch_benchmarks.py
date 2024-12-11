import torch
from typing import Any, Dict, List
import logging
from statistics import mean, stdev
import time
from benchmark_base import BenchmarkBase, BenchmarkResult


class PyTorchBenchmark(BenchmarkBase):
    def _initialize(self):
        """Initialize PyTorch-specific settings"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.manual_seed(self.config.seed)

    def benchmark_transformer_layer(
        self, batch_size: int, seq_len: int, hidden_size: int, num_heads: int
    ):
        """Benchmark transformer layer forward pass"""
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
        ).cuda()

        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda")

        def run_transformer():
            return encoder_layer(x)

        self.run_timed_operation(
            "transformer_layer",
            {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "num_heads": num_heads,
            },
            run_transformer,
        )

    def benchmark_lstm(self, batch_size: int, seq_len: int, hidden_size: int):
        """Benchmark LSTM forward and backward pass"""
        lstm = torch.nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        ).cuda()

        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda")
        target = torch.randn(batch_size, seq_len, hidden_size, device="cuda")

        def run_lstm():
            lstm.zero_grad()
            output, _ = lstm(x)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()

        self.run_timed_operation(
            "lstm",
            {"batch_size": batch_size, "seq_len": seq_len, "hidden_size": hidden_size},
            run_lstm,
        )

    def benchmark_mlp(self, batch_size: int, input_size: int, hidden_sizes: List[int]):
        """Benchmark MLP forward and backward pass"""
        layers = []
        prev_size = input_size

        for size in hidden_sizes:
            layers.extend(
                [
                    torch.nn.Linear(prev_size, size),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                ]
            )
            prev_size = size

        mlp = torch.nn.Sequential(*layers).cuda()
        x = torch.randn(batch_size, input_size, device="cuda")
        target = torch.randn(batch_size, hidden_sizes[-1], device="cuda")

        def run_mlp():
            mlp.zero_grad()
            output = mlp(x)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()

        self.run_timed_operation(
            "mlp",
            {
                "batch_size": batch_size,
                "input_size": input_size,
                "hidden_sizes": hidden_sizes,
            },
            run_mlp,
        )

    def benchmark_attention(self, batch_size: int, seq_len: int, hidden_size: int):
        """Benchmark attention mechanism"""
        scaling = float(hidden_size) ** -0.5
        q = torch.randn(batch_size, seq_len, hidden_size, device="cuda")
        k = torch.randn(batch_size, seq_len, hidden_size, device="cuda")
        v = torch.randn(batch_size, seq_len, hidden_size, device="cuda")

        def compute_attention():
            attn_weights = torch.bmm(q, k.transpose(1, 2)) * scaling
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            return torch.bmm(attn_weights, v)

        self.run_timed_operation(
            "attention",
            {"batch_size": batch_size, "seq_len": seq_len, "hidden_size": hidden_size},
            compute_attention,
        )

    def benchmark_embedding(self, batch_size: int, vocab_size: int, embedding_dim: int):
        """Benchmark embedding layer"""
        embedding = torch.nn.Embedding(vocab_size, embedding_dim).cuda()
        indices = torch.randint(0, vocab_size, (batch_size, 32), device="cuda")

        def run_embedding():
            return embedding(indices)

        self.run_timed_operation(
            "embedding",
            {
                "batch_size": batch_size,
                "vocab_size": vocab_size,
                "embedding_dim": embedding_dim,
            },
            run_embedding,
        )

    def benchmark_optimizer_step(self, num_params: int, optimizer: str):
        """Benchmark optimizer update step"""
        model = torch.nn.Linear(num_params, num_params).cuda()

        if optimizer == "adam":
            opt = torch.optim.Adam(model.parameters())
        elif optimizer == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        x = torch.randn(1, num_params, device="cuda")
        target = torch.randn(1, num_params, device="cuda")

        def optimization_step():
            opt.zero_grad()
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            opt.step()

        self.run_timed_operation(
            f"optimizer_{optimizer}",
            {"num_params": num_params, "optimizer": optimizer},
            optimization_step,
        )

    def benchmark_optimizer_step(self, num_params: int, optimizer: str):
        """Benchmark optimizer update step"""
        # Clear GPU cache before starting
        torch.cuda.empty_cache()

        # Use a more memory-efficient size for the model
        reduced_size = min(num_params, 512)  # Cap the size to prevent OOM
        model = torch.nn.Linear(reduced_size, reduced_size).cuda()

        if optimizer == "adam":
            opt = torch.optim.Adam(model.parameters())
        elif optimizer == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        x = torch.randn(1, reduced_size, device="cuda")
        target = torch.randn(1, reduced_size, device="cuda")

        def optimization_step():
            opt.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            opt.step()

        try:
            self.run_timed_operation(
                f"optimizer_{optimizer}",
                {
                    "num_params": reduced_size,  # Log the actual size used
                    "optimizer": optimizer,
                },
                optimization_step,
            )
        finally:
            # Cleanup
            del model, opt, x, target
            torch.cuda.empty_cache()

    def run_timed_operation(self, operation: str, parameters: Dict[str, Any], func):
        """Simple timing function for PyTorch operations with memory management"""
        times = []

        try:
            # Warmup phase
            for _ in range(self.config.warmup_iterations):
                func()
                torch.cuda.synchronize()

            # Timing phase
            for _ in range(self.config.test_iterations):
                # Clear cache before each iteration
                torch.cuda.empty_cache()

                start = time.perf_counter()
                func()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            # Store and log results
            result = BenchmarkResult(
                name=f"pytorch_{operation}",
                framework="pytorch",
                operation=operation,
                mean_time=mean(times),
                std_time=stdev(times),
                parameters=parameters,
            )
            self.results.append(result)
            logging.info(
                "PyTorch %s - Mean: %.3f ms, Std: %.3f ms, Params: %s",
                operation,
                result.mean_time * 1000,
                result.std_time * 1000,
                parameters,
            )
        finally:
            # Final cleanup
            torch.cuda.empty_cache()

    def benchmark_matmul(self, size: int):
        """Benchmark matrix multiplication"""
        a = torch.randn(size, size, device="cuda")
        b = torch.randn(size, size, device="cuda")

        def run_matmul():
            torch.matmul(a, b)

        self.run_timed_operation("matmul", {"size": size}, run_matmul)

    def benchmark_convolution(self, batch_size: int):
        """Benchmark 2D convolution"""
        in_channels, out_channels = 64, 128
        kernel_size = 3
        size = 32

        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=1).cuda()
        x = torch.randn(batch_size, in_channels, size, size, device="cuda")

        def run_conv():
            conv(x)

        self.run_timed_operation(
            "conv2d",
            {
                "batch_size": batch_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "size": size,
            },
            run_conv,
        )

    def benchmark_gradient(self, size: int):
        """Benchmark gradient computation"""
        x = torch.randn(size, size, requires_grad=True, device="cuda")

        def compute_grad():
            y = torch.sum(x**2)
            (grad,) = torch.autograd.grad(y, x)
            return grad

        self.run_timed_operation("gradient", {"size": size}, compute_grad)
