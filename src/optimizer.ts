import { Tensor, native } from "./tensor.js";
import { Parameter } from "./module.js";

export type ParameterValue = Tensor;

export class Optimizer {
    parameters: Parameter<Tensor>[];

    constructor(parameters: Parameter<Tensor>[]) {
        this.parameters = parameters;
    }
}

export class SGD extends Optimizer {
    lr: number;

    constructor(parameters: Parameter<Tensor>[], lr: number = 1.0) {
        super(parameters);
        this.lr = lr;
    }

    zeroGrad() {
        const ids = this.parameters.map(p => p.value._id);
        native.zeroGrad(ids);
    }

    step() {
        // Simple SGD: p = p - lr * grad
        for (const p of this.parameters) {
            const grad = p.value.grad;
            if (!grad) continue;
            const updated = p.value.sub(grad.mul(this.lr));
            p.update(updated as any);
        }
    }
}

export class Adam extends Optimizer {
    lr: number;
    beta1: number;
    beta2: number;
    eps: number;
    weightDecay: number;
    t: number = 0;

    constructor(
        parameters: Parameter<Tensor>[],
        { lr = 6e-4, beta1 = 0.9, beta2 = 0.95, eps = 1e-8, weightDecay = 0 } = {}
    ) {
        super(parameters);
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
    }

    zeroGrad() {
        const ids = this.parameters.map(p => p.value._id);
        native.zeroGrad(ids);
    }

    step(maxGradNorm: number = 0): number {
        this.t++;
        const ids = this.parameters.map(p => p.value._id);
        return native.clipAndStep(
            ids, this.lr, this.beta1, this.beta2, this.eps, this.weightDecay, this.t, maxGradNorm
        );
    }
}

export class GradScaler {
    private scale: number;
    private growthFactor: number;
    private backoffFactor: number;
    private growthInterval: number;
    private stepsSinceGrowth: number = 0;

    constructor({
        initScale = 65536.0,
        growthFactor = 2.0,
        backoffFactor = 0.5,
        growthInterval = 2000,
    } = {}) {
        this.scale = initScale;
        this.growthFactor = growthFactor;
        this.backoffFactor = backoffFactor;
        this.growthInterval = growthInterval;
    }

    getScale(): number {
        return this.scale;
    }

    scaleLoss(loss: Tensor): Tensor {
        return loss.mul(this.scale);
    }

    unscaleAndStep(optimizer: Adam, maxGradNorm: number = 0): { gradNorm: number; skipped: boolean } {
        const ids = optimizer.parameters.map(p => p.value._id);
        const invScale = 1.0 / this.scale;
        const foundInf = native.scaleGrads(ids, invScale);

        if (foundInf) {
            this.scale *= this.backoffFactor;
            this.stepsSinceGrowth = 0;
            optimizer.zeroGrad();
            return { gradNorm: 0, skipped: true };
        }

        const gradNorm = optimizer.step(maxGradNorm);
        this.stepsSinceGrowth++;
        if (this.stepsSinceGrowth >= this.growthInterval) {
            this.scale *= this.growthFactor;
            this.stepsSinceGrowth = 0;
        }
        return { gradNorm, skipped: false };
    }
}
