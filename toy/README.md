# Toy / Reference Implementations

This folder contains **pure-TypeScript** implementations of the core concepts used in `@mni-ml/framework`. These are educational reference implementations — simpler, slower, and meant for learning.

The production framework uses Rust N-API bindings for performance (CPU, CUDA, and WebGPU backends). The code here shows the same algorithms in plain TypeScript so you can understand what's happening under the hood.

## What's here

| File | Description |
|------|-------------|
| `scalar.ts` | Scalar value with autograd support |
| `scalar_functions.ts` | Differentiable scalar operations |
| `autodiff.ts` | Automatic differentiation engine (topological sort + backprop) |
| `operators.ts` | Basic mathematical operators |
| `tensor_data.ts` | Low-level tensor storage, shape, strides, indexing |
| `tensor_ops.ts` | Tensor map/zip/reduce/conv kernels |
| `tensor.ts` | Tensor class with autograd |
| `tensor_functions.ts` | Differentiable tensor operations |
| `module.ts` | Neural network module base class |
| `optimizer.ts` | SGD optimizer |
| `datasets.ts` | Simple dataset utilities |
| `fast_ops.ts` | Worker-thread parallelized tensor ops |
| `fast_ops_worker.ts` | Worker thread for fast_ops |
| `gpu_backend.ts` | WebGPU backend setup |
| `gpu_kernels.ts` | WGSL compute shader strings |
| `gpu_ops.ts` | GPU-accelerated tensor operations |

## Usage

These files are **not** part of the published package. They exist purely as a learning resource. If you want to understand how tensors, autograd, or neural networks work from scratch, start with `scalar.ts` and `autodiff.ts`, then move to `tensor_data.ts` and `tensor.ts`.
