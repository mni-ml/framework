import { Tensor, native } from "../../core/tensor.js";
import { Module, Parameter } from "./module.js";
import { randRange } from "./functional.js";

export class Embedding extends Module {
    weight!: Parameter<Tensor>;
    vocabSize: number;
    embedDim: number;

    constructor(vocabSize: number, embedDim: number) {
        super();
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;

        const bound = 1 / Math.sqrt(embedDim);
        this.weight = new Parameter(randRange([vocabSize, embedDim], -bound, bound));
    }

    forward(indices: number[][]): Tensor {
        const batch = indices.length;
        const seqLen = indices[0]!.length;
        const flat = indices.flat();
        const id = native.embeddingForward(this.weight.value._id, flat, batch, seqLen);
        return new Tensor(id);
    }
}

