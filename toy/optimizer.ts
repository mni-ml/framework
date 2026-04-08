import { Scalar } from "./scalar.js";
import { Parameter } from "./module.js";
import { Tensor } from "./tensor.js";
import { TensorData, createSharedStorage } from "./tensor_data.js";
import { TensorHistory } from "./tensor_functions.js";

export type ParameterValue = Scalar | Tensor;

export class Optimizer {
    parameters: Parameter<ParameterValue>[];

    constructor(parameters: Parameter<ParameterValue>[]) {
        this.parameters = parameters;
    }
}

export class SGD extends Optimizer {
    lr: number;

    constructor(parameters: Parameter<Scalar>[], lr: number = 1.0) {
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

            if (p.value instanceof Scalar) {
                const grad = p.value.derivative ?? 0;
                p.value.data -= this.lr * grad;
            } else if (p.value instanceof Tensor) {
                const grad = p.value.grad;
                if (!grad) continue;
                const valStorage = p.value.data.storage;
                const gradStorage = grad.data.storage;
                const size = p.value.size;
                const newStorage = createSharedStorage(size);
                for (let i = 0; i < size; i++) {
                    newStorage[i] = valStorage[i]! - this.lr * gradStorage[i]!;
                }
                const updated = new Tensor(new TensorData(newStorage, p.value.shape));
                updated.history = new TensorHistory();
                p.value = updated;
            }
        }
    }
}