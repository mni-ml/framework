import { Tensor } from '../../tstorch/dist/tensor.js';
import { Module, Parameter } from './module.js';

function randRange(shape: number[], min: number, max: number): Tensor {
  const size = shape.reduce((product, dim) => product * dim, 1);
  const data = Array.from({ length: size }, () => Math.random() * (max - min) + min);
  return Tensor.tensor(data, shape);
}

export class Linear extends Module {
  weight!: Parameter<Tensor>;
  bias!: Parameter<Tensor>;
  inFeatures: number;
  outFeatures: number;

  constructor(inFeatures: number, outFeatures: number) {
    super();
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;

    const bound = 1 / Math.sqrt(inFeatures);
    this.weight = new Parameter(randRange([inFeatures, outFeatures], -bound, bound));
    this.bias = new Parameter(randRange([outFeatures], -bound, bound));
  }

  forward(input: Tensor): Tensor {
    return input.matmul(this.weight.value).add(this.bias.value);
  }
}

export class ReLU extends Module {
  forward(input: Tensor): Tensor {
    return input.relu();
  }
}

export class Sigmoid extends Module {
  forward(input: Tensor): Tensor {
    return input.sigmoid();
  }
}

export class Tanh extends Module {
  forward(input: Tensor): Tensor {
    return input.mul(2).sigmoid().mul(2).sub(1);
  }
}

export function tile(input: Tensor, kernel: [number, number]): [Tensor, number, number] {
  const [batch, channel, height, width] = input.shape;
  const [kh, kw] = kernel;

  if (!batch || !channel || !height || !width) {
    throw new Error('pooling expects a 4D tensor');
  }
  if (height % kh !== 0 || width % kw !== 0) {
    throw new Error('input dimensions must be divisible by kernel dimensions');
  }

  const newHeight = height / kh;
  const newWidth = width / kw;
  const tiled = input
    .contiguous()
    .view(batch, channel, newHeight, kh, newWidth, kw)
    .permute(0, 1, 2, 4, 3, 5)
    .contiguous()
    .view(batch, channel, newHeight, newWidth, kh * kw);

  return [tiled, newHeight, newWidth];
}

export function avgpool2d(input: Tensor, kernel: [number, number] | number, maybeWidth?: number): Tensor {
  const actualKernel = Array.isArray(kernel) ? kernel : [kernel, maybeWidth ?? kernel];
  const [batch, channel] = input.shape;
  const [tiled, newHeight, newWidth] = tile(input, actualKernel);
  return tiled.mean(4).view(batch!, channel!, newHeight, newWidth);
}

export function maxpool2d(input: Tensor, kernel: [number, number] | number, maybeWidth?: number): Tensor {
  const actualKernel = Array.isArray(kernel) ? kernel : [kernel, maybeWidth ?? kernel];
  const [batch, channel] = input.shape;
  const [tiled, newHeight, newWidth] = tile(input, actualKernel);
  return tiled.max(4).view(batch!, channel!, newHeight, newWidth);
}

export function softmax(input: Tensor, dim: number): Tensor {
  const maxValues = input.max(dim);
  const expValues = input.sub(maxValues).exp();
  return expValues.mul(expValues.sum(dim).inv());
}

export function logsoftmax(input: Tensor, dim: number): Tensor {
  const maxValues = input.max(dim);
  const shifted = input.sub(maxValues);
  const logSumExp = shifted.exp().sum(dim).log();
  return shifted.sub(logSumExp);
}

export function dropout(input: Tensor, rate = 0.5, ignore = false): Tensor {
  if (ignore || rate === 0) {
    return input;
  }
  if (rate >= 1) {
    return Tensor.zeros(input.shape);
  }

  const mask = Tensor.rand(input.shape).gt(rate);
  return input.mul(mask).mul(1 / (1 - rate));
}

export function mseLoss(input: Tensor, target: Tensor): Tensor {
  const diff = input.sub(target);
  return diff.mul(diff).mean();
}

export function crossEntropyLoss(logits: Tensor, targets: Tensor): Tensor {
  const logProbs = logsoftmax(logits, logits.dims - 1);
  return targets.mul(logProbs).sum(logits.dims - 1).mean().neg();
}
