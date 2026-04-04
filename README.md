# TSTorch

TSTorch is a PyTorch-like Machine Learning framework in TypeScript + WebGPU, intended as a working library and educational resource.

Once completed, TSTorch will also allow runtime analysis, exposing the core execution mechanisms behind modern deep learning systems: autograd, graph capture, kernel fusion, and GPU memory planning.

## Project status

Active early development.

Features:
- Tensor operations
- Computation graph
- Backpropagation
- Autograd
- Multi-threaded accelerated operations
- GPU accelerated operations
- Matrix-multiplication using tensor-cores

Goals:

* predictable execution traces
* stable ONNX subset
* reproducible latency reporting across devices
* tooling for edge deployment decisions

## Steps to run demo

```bash
# Basic demos
pnpm run demo              # tensor training (default)
pnpm run demo scalar       # scalar autograd training

# Fast tensor training with CLI args (Task 3.5)
pnpm run demo fast --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05 --EPOCHS 500
pnpm run demo fast --DATASET all   # run all datasets
```

To run tests: `pnpm run test-tstorch`

## Task 3.5: Training Results

Configuration: `--BACKEND cpu --HIDDEN 100 --RATE 0.05`, 50 data points per dataset.

Element-wise ops use `fast_ops` (parallel worker threads when above threshold). MatMul uses CPU reference implementation.

### All Datasets (HIDDEN=100)

| Dataset | Epochs | Final Loss | Accuracy | Avg ms/epoch | Total Time |
|---------|--------|-----------|----------|-------------|------------|
| Simple  | 500    | 1.99      | 50/50    | 15.80ms     | 7.90s      |
| Diag    | 500    | 5.04      | 49/50    | 15.76ms     | 7.88s      |
| Split   | 500    | 20.96     | 48/50    | 16.00ms     | 8.00s      |
| Xor     | 500    | 23.02     | 45/50    | 15.57ms     | 7.78s      |
| Circle  | 1000   | 10.14     | 47/50    | 15.64ms     | 15.64s     |
| Spiral  | 1000   | 33.63     | 29/50    | 15.53ms     | 15.53s     |

**Average time per epoch: ~15.7ms** (3-layer MLP, 2→100→100→1, on CPU)

### Notes

- Spiral is the hardest dataset (non-linearly separable with rotational structure) and requires significantly more epochs/capacity to converge
- Simple and Diag converge to near-perfect accuracy within 500 epochs
- The fast_ops backend provides parallel element-wise operations via worker threads; matmul remains single-threaded CPU
- GPU backend (WebGPU) ops exist in the library but are not yet wired into the autodiff pipeline (requires async tensor execution)

## Acknowledgements
- [MiniTorch diy teaching library](https://minitorch.github.io/)
- [Good blog on autograd](https://mathblog.vercel.app/blog/autograd/)