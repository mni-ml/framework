import { Scalar } from "./scalar.js";
import { Parameter } from "./module.js";
import type { Tensor } from "./tensor.js";

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

            // Check for derivative (Scalar-like objects)
            if (p.value instanceof Scalar) {
                const grad = p.value.derivative ?? 0;
                p.value.data -= this.lr * grad;
            }
        }
    }
}