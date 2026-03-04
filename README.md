# TSTorch

TSTorch is a TypeScript implementation of PyTorch, intended as a working library and educational resource.

TSTorch exposes the core execution mechanisms behind modern deep learning systems: autograd, graph capture, kernel fusion, and GPU memory planning, using WebGPU as the primary execution target.

## Project status

Active early development.

Features:
- Tensor operations
- Computation graph
- Backpropogation
- Autograd
- Multi-threaded accelerated operations
- GPU accelerated operations
- Matrix-multiplication using tensor-cores

Goals:

* predictable execution traces
* stable ONNX subset
* reproducible latency reporting across devices
* tooling for edge deployment decisions

---

## Steps to run demo
To run demo: `pnpm run demo`
To run tests: `pnpm run test-tstorch`
