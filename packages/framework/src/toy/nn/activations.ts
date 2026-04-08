import { Tensor } from "../../core/tensor.js";
import { Module } from "./module.js";

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

