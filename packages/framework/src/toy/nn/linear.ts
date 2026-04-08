import { Tensor } from "../../core/tensor.js";
import { Module, Parameter } from "./module.js";
import { randRange } from "./functional.js";

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

