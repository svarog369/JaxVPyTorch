import jax
import jax.numpy as jnp
from typing import Any, Dict, List
import logging
from statistics import mean, stdev
import time
from jax import grad, jit, random
import numpy as np
from benchmark_base import BenchmarkBase, BenchmarkResult


class JAXBenchmark(BenchmarkBase):
    def _initialize(self):
        """Initialize JAX-specific settings"""
        np.random.seed(self.config.seed)
        self.key = random.PRNGKey(self.config.seed)

    def benchmark_transformer_layer(
        self, batch_size: int, seq_len: int, hidden_size: int, num_heads: int
    ):
        """Benchmark transformer layer forward pass"""

        def attention(query, key, value):
            d_k = query.shape[-1]
            scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / jnp.sqrt(d_k)
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.matmul(weights, value)

        def transformer_layer(x, weights):
            # Multi-head attention
            q = jnp.matmul(x, weights["query"])
            k = jnp.matmul(x, weights["key"])
            v = jnp.matmul(x, weights["value"])

            # Split heads
            q = q.reshape(batch_size, seq_len, num_heads, -1).transpose(0, 2, 1, 3)
            k = k.reshape(batch_size, seq_len, num_heads, -1).transpose(0, 2, 1, 3)
            v = v.reshape(batch_size, seq_len, num_heads, -1).transpose(0, 2, 1, 3)

            # Attention
            attn_output = attention(q, k, v)

            # Combine heads
            attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
                batch_size, seq_len, hidden_size
            )

            # FFN
            ffn_output = jax.nn.relu(jnp.matmul(attn_output, weights["ffn1"]))
            output = jnp.matmul(ffn_output, weights["ffn2"])

            return output

        # Initialize weights
        key1, key2, key3, key4, key5 = random.split(self.key, 5)
        weights = {
            "query": random.normal(key1, (hidden_size, hidden_size)),
            "key": random.normal(key2, (hidden_size, hidden_size)),
            "value": random.normal(key3, (hidden_size, hidden_size)),
            "ffn1": random.normal(key4, (hidden_size, hidden_size * 4)),
            "ffn2": random.normal(key5, (hidden_size * 4, hidden_size)),
        }

        x = random.normal(key1, (batch_size, seq_len, hidden_size))

        # JIT compile
        transformer_layer_jit = jit(lambda x: transformer_layer(x, weights))
        transformer_layer_jit(x)  # Compile

        self.run_timed_operation(
            "transformer_layer",
            {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "num_heads": num_heads,
            },
            transformer_layer_jit(x),
        )

    def benchmark_lstm(self, batch_size: int, seq_len: int, hidden_size: int):
        """Benchmark LSTM forward and backward pass"""

        def lstm_cell(carry, x):
            h_prev, c_prev = carry

            # Gates
            gates = jnp.concatenate([h_prev, x], axis=-1)
            gates = jnp.matmul(gates, weights["kernel"])

            # Split gates
            i, f, g, o = jnp.split(gates, 4, axis=-1)

            # Update cell state
            c = jnp.tanh(g) * jax.nn.sigmoid(i) + c_prev * jax.nn.sigmoid(f)
            h = jnp.tanh(c) * jax.nn.sigmoid(o)

            return (h, c), h

        # Initialize weights
        key1, key2 = random.split(self.key)
        weights = {"kernel": random.normal(key1, (hidden_size * 2, hidden_size * 4))}

        x = random.normal(key2, (batch_size, seq_len, hidden_size))
        h0 = jnp.zeros((batch_size, hidden_size))
        c0 = jnp.zeros((batch_size, hidden_size))

        # JIT compile
        lstm_scan = jit(lambda x: jax.lax.scan(lstm_cell, (h0, c0), x)[1])
        lstm_scan(x)  # Compile

        self.run_timed_operation(
            "lstm",
            {"batch_size": batch_size, "seq_len": seq_len, "hidden_size": hidden_size},
            lstm_scan(x),
        )

    def benchmark_mlp(self, batch_size: int, input_size: int, hidden_sizes: List[int]):
        """Benchmark MLP forward and backward pass"""

        def mlp(x, weights):
            for i in range(len(hidden_sizes)):
                x = jnp.matmul(x, weights[f"layer_{i}"])
                x = jax.nn.relu(x)
                x = jax.random.bernoulli(random.PRNGKey(0), 0.9, x.shape) * x  # dropout
            return x

        # Initialize weights
        weights = {}
        prev_size = input_size
        for i, size in enumerate(hidden_sizes):
            key = random.fold_in(self.key, i)
            weights[f"layer_{i}"] = random.normal(key, (prev_size, size))
            prev_size = size

        x = random.normal(self.key, (batch_size, input_size))

        # JIT compile
        mlp_jit = jit(lambda x: mlp(x, weights))
        mlp_jit(x)  # Compile

        self.run_timed_operation(
            "mlp",
            {
                "batch_size": batch_size,
                "input_size": input_size,
                "hidden_sizes": hidden_sizes,
            },
            mlp_jit(x),
        )

    def benchmark_attention(self, batch_size: int, seq_len: int, hidden_size: int):
        """Benchmark attention mechanism"""
        key1, key2, key3 = random.split(self.key, 3)
        q = random.normal(key1, (batch_size, seq_len, hidden_size))
        k = random.normal(key2, (batch_size, seq_len, hidden_size))
        v = random.normal(key3, (batch_size, seq_len, hidden_size))

        @jit
        def attention(q, k, v):
            scale = jnp.sqrt(hidden_size)
            scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / scale
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.matmul(weights, v)

        attention(q, k, v)  # Compile

        self.run_timed_operation(
            "attention",
            {"batch_size": batch_size, "seq_len": seq_len, "hidden_size": hidden_size},
            attention(q, k, v),
        )

    def benchmark_embedding(self, batch_size: int, vocab_size: int, embedding_dim: int):
        """Benchmark embedding layer"""
        key1, key2 = random.split(self.key)
        embedding_matrix = random.normal(key1, (vocab_size, embedding_dim))
        indices = random.randint(key2, (batch_size, 32), 0, vocab_size)

        @jit
        def embedding_lookup(indices):
            return jnp.take(embedding_matrix, indices, axis=0)

        embedding_lookup(indices)  # Compile

        self.run_timed_operation(
            "embedding",
            {
                "batch_size": batch_size,
                "vocab_size": vocab_size,
                "embedding_dim": embedding_dim,
            },
            embedding_lookup(indices),
        )

    def benchmark_optimizer_step(self, num_params: int, optimizer: str):
        """Benchmark optimizer update step"""
        key1, key2 = random.split(self.key)
        params = random.normal(key1, (num_params, num_params))
        x = random.normal(key2, (1, num_params))

        def loss_fn(params, x):
            output = jnp.matmul(x, params)
            return jnp.mean(output**2)

        if optimizer == "adam":
            opt_init, opt_update, get_params = jax.experimental.optimizers.adam(1e-3)
        elif optimizer == "sgd":
            opt_init, opt_update, get_params = jax.experimental.optimizers.sgd(0.01)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        opt_state = opt_init(params)

        @jit
        def update(opt_state, x):
            params = get_params(opt_state)
            grads = grad(loss_fn)(params, x)
            return opt_update(0, grads, opt_state)

        update(opt_state, x)  # Compile

        self.run_timed_operation(
            f"optimizer_{optimizer}",
            {"num_params": num_params, "optimizer": optimizer},
            update(opt_state, x),
        )

    def run_timed_operation(self, operation: str, parameters: Dict[str, Any], func):
        """Simple timing function for JAX operations"""
        times = []

        # Warmup phase
        for _ in range(self.config.warmup_iterations):
            func()

        # Timing phase
        for _ in range(self.config.test_iterations):
            start = time.perf_counter()
            func()
            times.append(time.perf_counter() - start)

        # Store and log results
        result = BenchmarkResult(
            name=f"jax_{operation}",
            framework="jax",
            operation=operation,
            mean_time=mean(times),
            std_time=stdev(times),
            parameters=parameters,
        )
        self.results.append(result)
        logging.info(
            "JAX %s - Mean: %.3f ms, Std: %.3f ms, Params: %s",
            operation,
            result.mean_time * 1000,
            result.std_time * 1000,
            parameters,
        )

    def benchmark_matmul(self, size: int):
        """Benchmark matrix multiplication"""
        key1, key2 = random.split(self.key)
        a = random.normal(key1, (size, size))
        b = random.normal(key2, (size, size))

        @jit
        def matmul(a, b):
            return jnp.matmul(a, b)

        # Compile once
        matmul(a, b)

        def run_matmul():
            matmul(a, b)

        self.run_timed_operation("matmul", {"size": size}, run_matmul)

    def benchmark_convolution(self, batch_size: int):
        """Benchmark 2D convolution"""
        in_channels, out_channels = 64, 128
        kernel_size = 3
        size = 32

        key1, key2 = random.split(self.key)
        kernel = random.normal(
            key1, (kernel_size, kernel_size, in_channels, out_channels)
        ) * jnp.sqrt(2.0 / (kernel_size * kernel_size * in_channels))
        x = random.normal(key2, (batch_size, size, size, in_channels))

        @jit
        def conv(x, kernel):
            return jax.lax.conv(x, kernel, window_strides=(1, 1), padding="SAME")

        # Compile once
        conv(x, kernel)

        def run_conv():
            conv(x, kernel)

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

    def benchmark_batch_norm(self, batch_size: int):
        """Benchmark batch normalization"""
        num_features = 64
        size = 32

        key = random.split(self.key)[0]
        x = random.normal(key, (batch_size, size, size, num_features))

        @jit
        def batchnorm(x):
            mean = jnp.mean(x, axis=(0, 1, 2), keepdims=True)
            var = jnp.var(x, axis=(0, 1, 2), keepdims=True)
            return (x - mean) / jnp.sqrt(var + 1e-5)

        # Compile once
        batchnorm(x)

        self.run_timed_operation(
            "batchnorm",
            {"batch_size": batch_size, "num_features": num_features},
            batchnorm(x),
        )

    def benchmark_gradient(self, size: int):
        """Benchmark gradient computation"""
        key = random.split(self.key)[0]
        x = random.normal(key, (size, size))

        @jit
        def compute_fn(x):
            return jnp.sum(x**2)

        grad_fn = jit(grad(compute_fn))

        # Compile once
        grad_fn(x)

        self.run_timed_operation("gradient", {"size": size}, grad_fn(x))
