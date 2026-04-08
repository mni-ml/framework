import { Tensor } from "../../core/tensor.js";
import { Module, Parameter } from "./module.js";

export class RMSNorm extends Module {
    weight!: Parameter<Tensor>;
    dimension: number;
    eps: number;

    constructor(dimension: number, eps: number = 1e-5) {
        super();
        this.dimension = dimension;
        this.eps = eps;

        const weight = Tensor.ones([dimension]);
        weight.setRequiresGrad(true);
        this.weight = new Parameter(weight);
    }

    forward(input: Tensor): Tensor {
        const meanSquares = input.mul(input).mean(input.dims - 1);
        const invRms = meanSquares.add(this.eps).log().mul(-0.5).exp();
        const normalized = input.mul(invRms.view(...meanSquares.shape, 1));
        const weightShape = [...Array(Math.max(0, input.dims - 1)).fill(1), this.dimension];

        return normalized.mul(this.weight.value.view(...weightShape));
    }
}

