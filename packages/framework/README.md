# @mni-ml/framework

A minimal machine learning library written in TypeScript. Implements core abstractions found in PyTorch -- autograd, tensors, modules, and training -- from scratch.

Built for learning and experimentation. Inspired by [minitorch](https://minitorch.github.io/).

## Install

```bash
npm install @mni-ml/framework
```

## What's included

- Scalar and tensor automatic differentiation (autograd)
- N-dimensional tensors backed by `Float64Array` with broadcasting and strided storage
- Element-wise, pairwise, and reduction ops, matrix multiplication, 1D and 2D convolutions
- Parallel CPU ops via worker threads
- Module system with automatic parameter registration, `train()`/`eval()` mode
- Layers: `Linear`, `Conv1d`, `Conv2d`, `Embedding`, `ReLU`, `Sigmoid`, `Tanh`
- Loss functions: `mseLoss`, `crossEntropyLoss`
- Functional ops: `softmax`, `logsoftmax`, `dropout`, `avgpool2d`, `maxpool2d`
- SGD optimizer
- Built-in 2D classification datasets

## Quick start

```typescript
import {
  Tensor, Linear, ReLU, SGD, mseLoss, Module, Parameter
} from "@mni-ml/framework";

class MLP extends Module {
  l1: Linear;
  l2: Linear;
  relu: ReLU;

  constructor() {
    super();
    this.l1 = new Linear(2, 10);
    this.relu = new ReLU();
    this.l2 = new Linear(10, 1);
  }

  forward(x: Tensor): Tensor {
    return this.l2.forward(this.relu.forward(this.l1.forward(x)));
  }
}

const model = new MLP();
const opt = new SGD(model.parameters(), 0.05);

for (let epoch = 0; epoch < 100; epoch++) {
  const x = Tensor.tensor([[0.1, 0.9], [0.8, 0.2]]);
  const target = Tensor.tensor([[1], [0]]);

  const pred = model.forward(x);
  const loss = mseLoss(pred, target);

  opt.zeroGrad();
  loss.backward();
  opt.step();
}
```

## API

### Tensor

```typescript
Tensor.tensor([[1, 2], [3, 4]])       // from nested arrays
Tensor.zeros([3, 3])                   // zeros
Tensor.ones([2, 4])                    // ones
Tensor.rand([2, 3])                    // uniform random

t.add(other)   t.sub(other)   t.mul(other)   // arithmetic
t.neg()        t.exp()        t.log()         // unary
t.sigmoid()    t.relu()                       // activations
t.matmul(other)                               // matrix multiply
t.conv1d(weight)  t.conv2d(weight)            // convolutions
t.sum(dim?)    t.mean(dim?)   t.max(dim)      // reductions
t.permute(...order)  t.view(...shape)         // reshaping
t.backward()                                  // backpropagation
```

### Modules

| Module | Description |
|--------|-------------|
| `Linear(in, out)` | Fully connected layer |
| `Conv1d(inCh, outCh, kernelW)` | 1D convolution |
| `Conv2d(inCh, outCh, [kH, kW])` | 2D convolution |
| `Embedding(numEmb, embDim)` | Lookup table with trainable weights |
| `ReLU` | Rectified linear unit |
| `Sigmoid` | Sigmoid activation |
| `Tanh` | Hyperbolic tangent activation |

### Loss functions

| Function | Use case |
|----------|----------|
| `mseLoss(pred, target)` | Regression |
| `crossEntropyLoss(pred, target)` | Classification |

### Functional

```typescript
softmax(input, dim)
logsoftmax(input, dim)
dropout(input, rate, ignore)
avgpool2d(input, [kH, kW])
maxpool2d(input, [kH, kW])
```

## License

MIT
