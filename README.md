# @mni-ml/framework

A TypeScript ML framework with Rust native backends (CPU, CUDA, WebGPU) providing autograd, tensor operations, and neural network training at GPU speed.

## Features

- **Automatic differentiation** -- full backward pass through an autograd tape
- **GPU acceleration** -- CUDA (NVIDIA) and WebGPU (Metal/Vulkan/DX12) backends
- **PyTorch-like API** -- familiar `Tensor`, `Module`, `Parameter`, optimizer classes
- **Comprehensive ops** -- elementwise, matmul, conv1d/conv2d, pooling, reductions, activations
- **Built-in modules** -- `Linear`, `Conv1d`, `Conv2d`, `Embedding`, `ReLU`, `Sigmoid`, `Tanh`
- **Optimizers** -- `SGD` and `Adam` (AdamW) with learning rate scheduling

## Installation

```bash
npm install @mni-ml/framework
```

### Building the native backend

The framework ships a Rust N-API backend. Choose a backend based on your hardware:

```bash
# CPU only (default, works everywhere)
cd native && cargo build --release --features cpu
cp target/release/libmni_framework_native.dylib mni-framework-native.darwin-arm64.node

# NVIDIA CUDA
cd native && cargo build --release --no-default-features --features cuda
cp target/release/libmni_framework_native.so mni-framework-native.linux-x64-gnu.node

# WebGPU (Metal on macOS, Vulkan on Linux/Windows)
cd native && cargo build --release --no-default-features --features webgpu
cp target/release/libmni_framework_native.dylib mni-framework-native.darwin-arm64.node
```

## Quick Start

```typescript
import { Tensor, Linear, Adam, Parameter, softmax, crossEntropyLoss } from '@mni-ml/framework';

// Create tensors
const x = Tensor.rand([32, 10]);
const targets = [[0], [1], [2], /* ... */];

// Build a model
const layer1 = new Linear(10, 64);
const layer2 = new Linear(64, 3);

// Forward pass
let h = layer1.forward(x).relu();
let logits = layer2.forward(h);
let loss = crossEntropyLoss(logits, targets);

// Backward pass
loss.backward();

// Optimize
const params = [...layer1.parameters(), ...layer2.parameters()];
const optimizer = new Adam(params, 0.001);
optimizer.step();
optimizer.zeroGrad();
```

## API Reference

### Tensor

```typescript
// Creation
Tensor.zeros([2, 3])           // zero-filled
Tensor.ones([2, 3])            // one-filled
Tensor.rand([2, 3])            // uniform [0, 1)
Tensor.randn([2, 3])           // normal distribution
Tensor.fromFloat32(data, shape) // from Float32Array

// Arithmetic (with autograd)
a.add(b)       a.add(2.0)     // addition
a.sub(b)                       // subtraction
a.mul(b)       a.mul(2.0)     // multiplication
a.div(b)       a.div(2.0)     // division
a.neg()                        // negation
a.exp()        a.log()        // exponentials
a.pow(2)                       // power

// Activations
a.relu()       a.sigmoid()

// Reductions
a.sum(dim)     a.sum()        // sum along dim or all
a.mean(dim)    a.mean()       // mean along dim or all
a.max(dim)                     // max along dim

// Comparisons (returns 0/1 tensor, no gradient)
a.lt(b)        a.gt(b)        a.eq(b)
a.isClose(b, tol)

// Layout
a.view(2, 3)                   // reshape
a.permute(1, 0)                // transpose
a.contiguous()                 // ensure contiguous memory

// Linear algebra
a.matmul(b)                    // matrix multiplication

// Convolution
a.conv1d(weight, stride, padding)
a.conv2d(weight, stride, padding)

// Utilities
a.clone()      a.detach()     // copy / detach from graph
a.toString()                   // debug string
a.backward()                   // run backward pass
a.setRequiresGrad(true)        // enable gradient tracking
```

### Neural Network Modules

```typescript
import { Linear, Conv1d, Conv2d, ReLU, Sigmoid, Embedding } from '@mni-ml/framework';

const linear = new Linear(inputSize, outputSize);
const conv1d = new Conv1d(inChannels, outChannels, kernelSize, stride, padding);
const conv2d = new Conv2d(inChannels, outChannels, kernelSize, stride, padding);
const embedding = new Embedding(vocabSize, embeddingDim);

// Use in forward pass
const out = linear.forward(input);
```

### Functional Operations

```typescript
import { softmax, gelu, layerNorm, crossEntropyLoss, dropout,
         avgpool2d, maxpool2d, tile } from '@mni-ml/framework';

const sm = softmax(logits, dim);
const g = gelu(x);
const ln = layerNorm(x, gamma, beta, eps);
const loss = crossEntropyLoss(logits, targets);
const dropped = dropout(x, rate, training);
const pooled = avgpool2d(x, kernelH, kernelW);
const maxPooled = maxpool2d(x, kernelH, kernelW);
const tiled = tile(x, [2, 1]);
```

### Optimizers

```typescript
import { Adam, SGD } from '@mni-ml/framework';

const optimizer = new Adam(parameters, lr, beta1, beta2, eps, weightDecay);
// or
const optimizer = new SGD(parameters, lr);

optimizer.step();      // update parameters
optimizer.zeroGrad();  // clear gradients
```

## Backend Architecture

```
TypeScript API (tensor.ts, nn.ts, optimizer.ts)
    │
    └─→ N-API Bridge (lib.rs)
            │
            ├─→ CPU Backend (Vec<f32>, pure Rust)
            ├─→ CUDA Backend (cudarc + .cu kernels)
            └─→ WebGPU Backend (wgpu + .wgsl shaders)
```

All three backends share the same autograd tape and tensor store. Feature flags are mutually exclusive at compile time:

- `cpu` -- default, no GPU required
- `cuda` -- NVIDIA GPU via CUDA
- `webgpu` -- any GPU via wgpu (Metal, Vulkan, DX12)

## License

MIT
