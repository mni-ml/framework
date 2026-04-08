import { Tensor } from "../../core/tensor.js";
import { Module, Parameter } from "./module.js";
import { randRange } from "./functional.js";

export class Conv1dModule extends Module {
    weight!: Parameter<Tensor>;
    bias!: Parameter<Tensor>;
    inChannels: number;
    outChannels: number;
    kernelWidth: number;

    constructor(inChannels: number, outChannels: number, kernelWidth: number) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelWidth = kernelWidth;

        const bound = 1 / Math.sqrt(inChannels * kernelWidth);
        this.weight = new Parameter(randRange([outChannels, inChannels, kernelWidth], -bound, bound));
        this.bias = new Parameter(randRange([outChannels], -bound, bound));
    }

    forward(input: Tensor): Tensor {
        return input.conv1d(this.weight.value).add(this.bias.value.view(1, this.outChannels, 1));
    }
}

export class Conv2dModule extends Module {
    weight!: Parameter<Tensor>;
    bias!: Parameter<Tensor>;
    inChannels: number;
    outChannels: number;
    kernelSize: [number, number];

    constructor(inChannels: number, outChannels: number, kernelSize: [number, number]) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;

        const [kernelHeight, kernelWidth] = kernelSize;
        const bound = 1 / Math.sqrt(inChannels * kernelHeight * kernelWidth);
        this.weight = new Parameter(
            randRange([outChannels, inChannels, kernelHeight, kernelWidth], -bound, bound),
        );
        this.bias = new Parameter(randRange([outChannels], -bound, bound));
    }

    forward(input: Tensor): Tensor {
        return input
            .conv2d(this.weight.value)
            .add(this.bias.value.view(1, this.outChannels, 1, 1));
    }
}

