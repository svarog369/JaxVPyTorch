import jax
import jax.numpy as jnp
import optax
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

        # Compile once
        transformer_layer_jit(x)

        def run_transformer():
            transformer_layer_jit(x)

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

        def lstm_cell(carry, x_t):
            h_prev, c_prev = carry

            # Ensure x_t has shape (batch_size, hidden_size)
            if len(x_t.shape) == 1:
                x_t = jnp.expand_dims(x_t, 0)

            # Ensure h_prev and c_prev match batch size
            if h_prev.shape[0] != x_t.shape[0]:
                h_prev = jnp.repeat(h_prev, x_t.shape[0], axis=0)
                c_prev = jnp.repeat(c_prev, x_t.shape[0], axis=0)

            # Gates
            gates = jnp.concatenate([h_prev, x_t], axis=-1)
            gates = jnp.matmul(gates, weights["kernel"])

            # Split gates
            gates = jnp.split(gates, 4, axis=-1)
            i, f, g, o = gates

            # Apply gate activations
            i = jax.nn.sigmoid(i)
            f = jax.nn.sigmoid(f)
            g = jnp.tanh(g)
            o = jax.nn.sigmoid(o)

            # Update cell state
            c = f * c_prev + i * g
            h = o * jnp.tanh(c)

            return (h, c), h

        # Initialize weights with proper scaling
        key1, key2 = random.split(self.key)
        input_size = hidden_size
        weights = {
            "kernel": random.normal(key1, (input_size + hidden_size, hidden_size * 4))
            / jnp.sqrt(input_size + hidden_size)
        }

        # Create input sequence
        x = random.normal(key2, (seq_len, batch_size, hidden_size))

        # Initialize hidden states properly
        h0 = jnp.zeros((batch_size, hidden_size))
        c0 = jnp.zeros((batch_size, hidden_size))

        # JIT compile the scan
        @jit
        def lstm_forward(x):
            return jax.lax.scan(lstm_cell, (h0, c0), x)[1]

        # Compile once
        lstm_forward(x)

        def run_lstm():
            lstm_forward(x)

        self.run_timed_operation(
            "lstm",
            {"batch_size": batch_size, "seq_len": seq_len, "hidden_size": hidden_size},
            run_lstm,
        )

    def benchmark_mlp(self, batch_size: int, input_size: int, hidden_sizes: List[int]):
        """Benchmark MLP forward and backward pass"""

        def mlp(x, weights):
            activation = x
            for i in range(len(hidden_sizes)):
                activation = jnp.matmul(activation, weights[f"layer_{i}"])
                activation = jax.nn.relu(activation)
                # Apply dropout using JAX's random key
                key = random.fold_in(self.key, i)
                mask = random.bernoulli(key, 0.9, activation.shape)
                activation = mask * activation
            return activation

        # Initialize weights with proper scaling
        weights = {}
        prev_size = input_size
        for i, size in enumerate(hidden_sizes):
            key = random.fold_in(self.key, i)
            scale = jnp.sqrt(2.0 / prev_size)  # He initialization
            weights[f"layer_{i}"] = random.normal(key, (prev_size, size)) * scale
            prev_size = size

        # Create input data
        x = random.normal(self.key, (batch_size, input_size))

        # JIT compile
        mlp_jit = jit(lambda x: mlp(x, weights))

        # Compile once
        mlp_jit(x)

        def run_mlp():
            mlp_jit(x)

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

        # Compile once
        attention(q, k, v)

        def run_attention():
            attention(q, k, v)

        self.run_timed_operation(
            "attention",
            {"batch_size": batch_size, "seq_len": seq_len, "hidden_size": hidden_size},
            run_attention,
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

        def run_embedding_lookup():
            embedding_lookup(indices)

        self.run_timed_operation(
            "embedding",
            {
                "batch_size": batch_size,
                "vocab_size": vocab_size,
                "embedding_dim": embedding_dim,
            },
            run_embedding_lookup,
        )

    def benchmark_optimizer_step(self, num_params: int, optimizer: str):
        """Benchmark optimizer update step with memory-efficient implementation"""
        # Cap the parameter size to prevent OOM
        max_params = 1000  # Limit to prevent excessive memory usage
        reduced_size = min(num_params, max_params)

        key1, key2 = random.split(self.key)

        # Use a more memory-efficient parameter shape (n×k) instead of (n×n)
        k = min(reduced_size, 100)  # Further reduce second dimension
        params = random.normal(key1, (reduced_size, k))
        x = random.normal(key2, (1, reduced_size))

        def loss_fn(params, x):
            # Use a more memory-efficient computation
            output = jnp.matmul(x, params)  # Results in (1, k) matrix
            return jnp.mean(output**2)

        if optimizer == "adam":
            tx = optax.adam(learning_rate=1e-3)
        elif optimizer == "sgd":
            tx = optax.sgd(learning_rate=0.01)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # Initialize optimizer state
        opt_state = tx.init(params)

        # Calculate gradients
        @jit
        def compute_step(params, opt_state, x):
            loss_value, grads = jax.value_and_grad(loss_fn)(params, x)
            updates, new_opt_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss_value

        # Compile once
        compute_step(params, opt_state, x)

        def run_step():
            nonlocal params, opt_state
            params, opt_state, _ = compute_step(params, opt_state, x)

        try:
            self.run_timed_operation(
                f"optimizer_{optimizer}",
                {
                    "num_params": reduced_size,  # Log the actual size used
                    "optimizer": optimizer,
                    "k_dim": k,  # Log the reduced second dimension
                },
                run_step,
            )
        finally:
            # Cleanup
            params = None
            opt_state = None
            jax.clear_caches()  # Clear any cached computations

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

        # JAX expects kernels in (out_channels, in_channels, height, width) format
        kernel = random.normal(
            key1, (out_channels, in_channels, kernel_size, kernel_size)
        ) * jnp.sqrt(2.0 / (kernel_size * kernel_size * in_channels))

        # JAX expects input in (batch_size, in_channels, height, width) format
        x = random.normal(key2, (batch_size, in_channels, size, size))

        @jit
        def conv(x, kernel):
            return jax.lax.conv_general_dilated(
                x,
                kernel,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),  # Specify dimension format
            )

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

        def run_grad():
            grad_fn(x)

        self.run_timed_operation("gradient", {"size": size}, run_grad)
