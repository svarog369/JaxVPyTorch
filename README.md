# PyTorch vs JAX Benchmark Suite

A comprehensive benchmarking suite comparing PyTorch and JAX performance on GPU for various deep learning operations and architectures.

## Features

### Basic Operations
- Matrix Multiplication
- 2D Convolution (with NCHW format)
- Gradient Computation

### Neural Network Components
- Transformer Layers
  - Multi-head Attention
  - Feed-forward Networks
  - Different head configurations
  - Memory-efficient implementation
- LSTM Networks
  - Various sequence lengths
  - Forward and backward passes
  - Proper shape handling for batched sequences
- Multi-Layer Perceptrons (MLPs)
  - Different architectures
  - Dropout and ReLU activations
  - Memory-efficient parameter handling
- Attention Mechanisms
  - Scaled dot-product attention
  - Different sequence lengths
  - Optimized tensor operations
- Embedding Layers
  - Various vocabulary sizes
  - Different embedding dimensions
- Optimizer Performance
  - Adam (using optax for JAX)
  - SGD
  - Memory-efficient parameter handling
  - Configurable batch sizes

## Requirements

- Python 3.8+
- CUDA-capable GPU
- CUDA drivers and toolkit
- Dependencies listed in `requirements.txt`:
  ```
  torch>=2.0.0
  jax[cuda]>=0.4.1
  optax  # Required for JAX optimizers
  numpy>=1.21.0
  matplotlib>=3.4.0
  typing-extensions>=4.0.0
  optax>=0.2.3
  ```

## Memory Considerations

The benchmark suite is designed to be memory-efficient and includes several optimizations:

1. Parameter Size Limitations:
   - Matrix operations use capped sizes
   - Optimizer benchmarks use rectangular matrices instead of square
   - Automatic size reduction for large parameter counts

2. Memory Management:
   - Explicit cleanup after benchmarks
   - Cache clearing for JAX operations
   - GPU memory monitoring
   - Proper tensor cleanup

## Installation

1. Clone the repository:
```bash
git clone https://github.com/svarog369/JaxVPyTorch.git
cd jaxvpytorch
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify CUDA availability:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import jax; print('JAX devices:', jax.devices())"
```

## Project Structure

```
pytorchvsjax/
├── benchmark_base.py     # Base classes and utilities
├── pytorch_benchmarks.py # PyTorch implementations
├── jax_benchmarks.py     # JAX implementations
├── main.py              # Main script to run benchmarks
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Usage

### Running All Benchmarks

```bash
python main.py
```

This will:
1. Run all benchmarks with the default configuration
2. Generate detailed logs in `benchmark_results.log`
3. Save structured results in `benchmark_results.json`
4. Create performance comparison plots as `benchmark_*.png`

## Monitoring

### GPU Utilization
Monitor GPU usage during benchmarks:
- Linux: `nvidia-smi -l 1`
- Windows: Task Manager's Performance tab
- Detailed monitoring: `nvidia-smi dmon`

### Memory Usage
Some operations with large sizes may require substantial GPU memory. Monitor memory usage and adjust configurations accordingly.

## Output Files

1. `benchmark_results.log`:
   - Detailed timing information
   - Configuration details
   - Error messages and warnings

2. `benchmark_results.json`:
   ```json
   {
     "config": {
       "batch_sizes": [...],
       "matrix_sizes": [...],
       ...
     },
     "results": [
       {
         "name": "pytorch_matmul",
         "framework": "pytorch",
         "operation": "matmul",
         "mean_time_ms": 1.234,
         "std_time_ms": 0.123,
         "parameters": {
           "size": 1024
         }
       },
       ...
     ]
   }
   ```

3. `benchmark_*.png`:
   - Performance comparison plots
   - One plot per operation type
   - PyTorch vs JAX timing comparisons

## Extending the Benchmark Suite

### Adding New Operations

1. Add the operation to `BenchmarkBase` in `benchmark_base.py`:
```python
def benchmark_new_operation(self, param1, param2):
    """Benchmark new operation"""
    raise NotImplementedError
```

2. Implement in both `PyTorchBenchmark` and `JAXBenchmark` classes
3. Add to the benchmark loop in `main.py`

### Adding New Configurations

Modify `BenchmarkConfig` and `NNConfig` in `benchmark_base.py` to add new configuration parameters.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

