import { Tensor, native } from "../core/tensor.js";
import { Parameter } from "./nn/module.js";

type ScalarLike = {
    data: number;
    derivative: number | null;
};

function isTensorValue(value: unknown): value is Tensor {
    return value instanceof Tensor;
}

function isScalarValue(value: unknown): value is ScalarLike {
    return typeof value === "object"
        && value !== null
        && "data" in value
        && "derivative" in value;
}

export type ParameterValue = Tensor | ScalarLike;

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

    zeroGrad(): void {
        const tensorIds: number[] = [];

        for (const parameter of this.parameters) {
            if (isTensorValue(parameter.value)) {
                tensorIds.push(parameter.value._id);
            } else if (isScalarValue(parameter.value)) {
                parameter.value.derivative = null;
            }
        }

        if (tensorIds.length > 0) {
            native.zeroGrad(tensorIds);
        }
    }

    step(): void {
        for (const parameter of this.parameters) {
            if (isTensorValue(parameter.value)) {
                const grad = parameter.value.grad;
                if (!grad) {
                    continue;
                }

                parameter.update(parameter.value.sub(grad.mul(this.lr)));
                continue;
            }

            if (isScalarValue(parameter.value) && parameter.value.derivative != null) {
                parameter.value.data -= parameter.value.derivative * this.lr;
                parameter.value.derivative = null;
            }
        }
    }
}

export class Adam extends Optimizer {
    lr: number;
    beta1: number;
    beta2: number;
    eps: number;
    weightDecay: number;
    t = 0;

    constructor(
        parameters: Parameter<ParameterValue>[],
        { lr = 6e-4, beta1 = 0.9, beta2 = 0.95, eps = 1e-8, weightDecay = 0 } = {},
    ) {
        super(parameters);
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
    }

    zeroGrad(): void {
        const ids = this.parameters
            .map(parameter => (isTensorValue(parameter.value) ? parameter.value._id : null))
            .filter((id): id is number => id != null);
        native.zeroGrad(ids);
    }

    step(): void {
        this.t++;
        const ids = this.parameters
            .map(parameter => (isTensorValue(parameter.value) ? parameter.value._id : null))
            .filter((id): id is number => id != null);
        native.adamwStep(ids, this.lr, this.beta1, this.beta2, this.eps, this.weightDecay, this.t);
    }
}

