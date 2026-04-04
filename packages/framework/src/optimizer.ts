import { Scalar } from "./scalar.js";
import { Parameter } from "./module.js";
import { Tensor } from "./tensor.js";
import { TensorData } from "./tensor_data.js";

export type ParameterValue = Scalar | Tensor;

export class Optimizer {
    parameters: Parameter<ParameterValue>[];

    constructor(parameters: Parameter<ParameterValue>[]) {
        this.parameters = parameters;
    }
}

export class SGD extends Optimizer {
    lr: number;

    constructor(parameters: Parameter<ParameterValue>[], lr: number = 1.0) {
        super(parameters);
        this.lr = lr;
    }

    zeroGrad() {
        for (let p of this.parameters) {
            if (!p.value || typeof p.value !== 'object') {
                continue;
            }
            if ("derivative" in p.value) { 
                if (p.value.derivative !== null && p.value.derivative !== undefined) {
                    p.value.derivative = 0;
                }
            }
            if ("grad" in p.value) { 
                if (p.value.grad !== null && p.value.grad !== undefined) {
                    p.value.grad = null;
                }
            }
        }
    }

    step() {
        for (let p of this.parameters) {
            if (!p.value || typeof p.value !== 'object') {
                continue;
            }

            if (p.value instanceof Tensor) {
                const rawGrad = p.value.grad;
                if (rawGrad) {
                    // Ensure contiguous layout before raw storage access —
                    // gradients from permute backward etc. may have non-trivial strides.
                    const grad = rawGrad.contiguous();
                    const val = p.value.contiguous();
                    const valStorage = val.data.storage;
                    const gradStorage = grad.data.storage;
                    const newStorage = new Float64Array(val.size);
                    for (let i = 0; i < val.size; i++) {
                        newStorage[i] = valStorage[i]! - this.lr * gradStorage[i]!;
                    }
                    p.update(new Tensor(new TensorData(newStorage, [...val.shape])) as any);
                }
            } else if (p.value instanceof Scalar) {
                const grad = p.value.derivative ?? 0;
                p.value.data -= this.lr * grad;
            }
        }
    }
}