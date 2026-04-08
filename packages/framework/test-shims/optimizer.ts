import { Scalar } from '../../tstorch/dist/scalar.js';
import { Tensor } from '../../tstorch/dist/tensor.js';
import { TensorData } from '../../tstorch/dist/tensor_data.js';
import { Parameter } from './module.js';

export type ParameterValue = Tensor | Scalar;

export class Optimizer {
  parameters: Parameter<ParameterValue>[];

  constructor(parameters: Parameter<ParameterValue>[]) {
    this.parameters = parameters;
  }
}

export class SGD extends Optimizer {
  lr: number;

  constructor(parameters: Parameter<ParameterValue>[], lr = 1.0) {
    super(parameters);
    this.lr = lr;
  }

  zeroGrad(): void {
    for (const parameter of this.parameters) {
      if (parameter.value instanceof Scalar) {
        parameter.value.derivative = 0;
      } else if (parameter.value instanceof Tensor) {
        parameter.value.zero_grad_();
      }
    }
  }

  step(): void {
    for (const parameter of this.parameters) {
      if (parameter.value instanceof Tensor) {
        const grad = parameter.value.grad;
        if (!grad) {
          continue;
        }

        const nextStorage = new Float64Array(parameter.value.size);
        const valueStorage = parameter.value.data.storage;
        const gradStorage = grad.data.storage;

        for (let i = 0; i < parameter.value.size; i++) {
          nextStorage[i] = valueStorage[i]! - this.lr * gradStorage[i]!;
        }

        parameter.update(new Tensor(new TensorData(nextStorage, [...parameter.value.shape])) as ParameterValue);
      } else if (parameter.value instanceof Scalar) {
        parameter.value.data -= this.lr * (parameter.value.derivative ?? 0);
      }
    }
  }
}

export class Adam extends Optimizer {
  lr: number;

  constructor(parameters: Parameter<ParameterValue>[], { lr = 6e-4 } = {}) {
    super(parameters);
    this.lr = lr;
  }

  zeroGrad(): void {
    for (const parameter of this.parameters) {
      if (parameter.value instanceof Scalar) {
        parameter.value.derivative = 0;
      } else if (parameter.value instanceof Tensor) {
        parameter.value.zero_grad_();
      }
    }
  }

  step(): void {
    throw new Error('Adam is not implemented in the pure unit-test shim.');
  }
}
