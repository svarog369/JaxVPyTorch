import logging
from benchmark_base import BenchmarkConfig, ResultManager
from pytorch_benchmarks import PyTorchBenchmark
from jax_benchmarks import JAXBenchmark


def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("benchmark_results.log"),
            logging.StreamHandler(),
        ],
    )


def run_benchmarks():
    """Run all benchmarks"""
    setup_logging()
    config = BenchmarkConfig()

    # Initialize benchmarks
    pytorch_bench = PyTorchBenchmark(config)
    jax_bench = JAXBenchmark(config)

    # Run benchmarks for both frameworks
    for bench in [pytorch_bench, jax_bench]:
        # Basic operations benchmarks
        for size in config.matrix_sizes:
            bench.benchmark_matmul(size)

        for batch_size in config.batch_sizes:
            bench.benchmark_convolution(batch_size)

        for size in config.matrix_sizes:
            bench.benchmark_gradient(size)

        # Neural network specific benchmarks
        for batch_size in config.batch_sizes:
            # Transformer benchmarks
            for seq_len in config.nn_config.sequence_lengths:
                for hidden_size in config.nn_config.hidden_sizes:
                    for num_heads in config.nn_config.num_heads:
                        if hidden_size % num_heads == 0:  # Valid head configuration
                            bench.benchmark_transformer_layer(
                                batch_size, seq_len, hidden_size, num_heads
                            )

            # LSTM benchmarks
            for seq_len in config.nn_config.sequence_lengths:
                for hidden_size in config.nn_config.hidden_sizes:
                    bench.benchmark_lstm(batch_size, seq_len, hidden_size)

            # MLP benchmarks
            input_sizes = [32, 64, 128, 256]
            for input_size in input_sizes:
                bench.benchmark_mlp(
                    batch_size, input_size, config.nn_config.hidden_sizes
                )

            # Attention benchmarks
            for seq_len in config.nn_config.sequence_lengths:
                for hidden_size in config.nn_config.hidden_sizes:
                    bench.benchmark_attention(batch_size, seq_len, hidden_size)

            # Embedding benchmarks
            vocab_sizes = [1000, 5000, 10000, 50000]
            for vocab_size in vocab_sizes:
                for embed_dim in config.nn_config.embedding_dims:
                    bench.benchmark_embedding(batch_size, vocab_size, embed_dim)

        # Optimizer benchmarks
        param_sizes = [1000, 10000, 100000]
        optimizers = ["adam", "sgd"]
        for num_params in param_sizes:
            for opt in optimizers:
                bench.benchmark_optimizer_step(num_params, opt)

    # Combine and save results
    result_manager = ResultManager()
    result_manager.add_results(pytorch_bench.results)
    result_manager.add_results(jax_bench.results)
    result_manager.save_results(config)
    result_manager.plot_results()


if __name__ == "__main__":
    run_benchmarks()
