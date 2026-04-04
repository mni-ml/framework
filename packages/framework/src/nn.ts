import { Tensor } from "./tensor.js"
import { TensorData, shapeProduct, type Shape } from "./tensor_data.js"
import { TensorFunction, TensorContext } from "./tensor_functions.js"
import { Module, Parameter } from "./module.js"

// ============================================================
// Modules
// ============================================================

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

export class Conv1d extends Module {
    weight!: Parameter<Tensor>;
    inChannels: number;
    outChannels: number;
    kernelWidth: number;

    constructor(inChannels: number, outChannels: number, kernelWidth: number) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelWidth = kernelWidth;
        const bound = 1 / Math.sqrt(inChannels * kernelWidth);
        this.weight = new Parameter(
            randRange([outChannels, inChannels, kernelWidth], -bound, bound)
        );
    }

    forward(input: Tensor): Tensor {
        return input.conv1d(this.weight.value);
    }
}

export class Conv2d extends Module {
    weight!: Parameter<Tensor>;
    inChannels: number;
    outChannels: number;
    kernelHeight: number;
    kernelWidth: number;

    constructor(inChannels: number, outChannels: number, kernelSize: [number, number]) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelHeight = kernelSize[0];
        this.kernelWidth = kernelSize[1];
        const bound = 1 / Math.sqrt(inChannels * this.kernelHeight * this.kernelWidth);
        this.weight = new Parameter(
            randRange([outChannels, inChannels, this.kernelHeight, this.kernelWidth], -bound, bound)
        );
    }

    forward(input: Tensor): Tensor {
        return input.conv2d(this.weight.value);
    }
}

/**
 * Creates a TensorFunction for embedding lookup with efficient O(B*S*D) backward
 * instead of the naive O(B*S*V*D) one-hot matmul approach.
 */
function EmbeddingLookupFn(
    indices: number[][], batch: number, seqLen: number, numEmb: number, embDim: number
): typeof TensorFunction {
    function validateIndices() {
        for (let b = 0; b < batch; b++) {
            for (let s = 0; s < seqLen; s++) {
                const idx = indices[b]![s]!;
                if (idx < 0 || idx >= numEmb) {
                    throw new RangeError(
                        `Embedding index ${idx} out of range [0, ${numEmb}) at position [${b}, ${s}]`
                    );
                }
            }
        }
    }

    return class EmbeddingLookup extends TensorFunction {
        static forward(ctx: TensorContext, weight: Tensor): Tensor {
            validateIndices();
            ctx.saveForBackward(weight);
            const wStorage = weight.data.storage;
            const storage = new Float64Array(batch * seqLen * embDim);
            let offset = 0;
            for (let b = 0; b < batch; b++) {
                for (let s = 0; s < seqLen; s++) {
                    const wBase = indices[b]![s]! * embDim;
                    for (let d = 0; d < embDim; d++) {
                        storage[offset++] = wStorage[wBase + d]!;
                    }
                }
            }
            return new Tensor(new TensorData(storage, [batch, seqLen, embDim]));
        }

        static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
            const go = gradOutput.contiguous();
            const goStorage = go.data.storage;
            const gradWeight = new Float64Array(numEmb * embDim);
            let offset = 0;
            for (let b = 0; b < batch; b++) {
                for (let s = 0; s < seqLen; s++) {
                    const wBase = indices[b]![s]! * embDim;
                    for (let d = 0; d < embDim; d++) {
                        gradWeight[wBase + d] = gradWeight[wBase + d]! + goStorage[offset++]!;
                    }
                }
            }
            return [new Tensor(new TensorData(gradWeight, [numEmb, embDim]))];
        }
    };
}

export class Embedding extends Module {
    weight!: Parameter<Tensor>;
    numEmbeddings: number;
    embeddingDim: number;

    constructor(numEmbeddings: number, embeddingDim: number, weights?: Tensor) {
        super();
        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;
        this.weight = new Parameter(weights ?? randRange([numEmbeddings, embeddingDim], -0.1, 0.1));
    }

    static fromPretrained(weights: Tensor): Embedding {
        const [numEmb, embDim] = weights.shape;
        return new Embedding(numEmb!, embDim!, weights);
    }

    forward(indices: number[][]): Tensor {
        const batch = indices.length;
        const seqLen = indices[0]!.length;
        const fn = EmbeddingLookupFn(indices, batch, seqLen, this.numEmbeddings, this.embeddingDim);
        return Tensor.apply(fn, this.weight.value);
    }
}

// ============================================================
// Loss functions
// ============================================================

export function mseLoss(input: Tensor, target: Tensor): Tensor {
    const diff = input.sub(target);
    return diff.mul(diff).mean();
}

export function crossEntropyLoss(input: Tensor, target: Tensor): Tensor {
    const lsm = logsoftmax(input, input.dims - 1);
    return lsm.mul(target).neg().sum(input.dims - 1).mean();
}

// ============================================================
// Initialization helpers
// ============================================================

export function randRange(shape: Shape, low: number, high: number): Tensor {
    const size = shapeProduct(shape);
    const storage = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        storage[i] = Math.random() * (high - low) + low;
    }
    return new Tensor(new TensorData(storage, [...shape]));
}

// ============================================================
// Functional NN ops
// ============================================================

/**
 * Reshape an image tensor for 2D pooling.
 *
 * input:  batch x channel x height x width
 * kernel: [kh, kw]
 * returns: tensor of shape [batch, channel, newHeight, newWidth, kh*kw],
 *          plus the newHeight and newWidth values.
 */
export function tile(input: Tensor, kernel: [number, number]): [Tensor, number, number] {
    const [batch, channel, height, width] = input.shape;
    const [kh, kw] = kernel;

    if (!batch) throw new Error("input batch undefined");
    if (!channel) throw new Error("input channel undefined");
    if (!height) throw new Error("input height undefined");
    if (!width) throw new Error("input width undefined");

    if (height % kh !== 0) {
        throw new Error("input height must be divisible by kernel height");
    }
    if (width % kw !== 0) {
        throw new Error("input width must be divisible by kernel width");
    }

    const newHeight = height / kh;
    const newWidth = width / kw;

    const tiled = input.contiguous()
        .view(batch, channel, newHeight, kh, newWidth, kw)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(batch, channel, newHeight, newWidth, kh * kw);

    return [tiled, newHeight, newWidth];
}

export function avgpool2d(input: Tensor, kernel: [number, number]): Tensor {
    const [batch, channel] = input.shape;
    const [tiled, newHeight, newWidth] = tile(input, kernel);
    return tiled.mean(4).view(batch!, channel!, newHeight, newWidth);
}

export function maxpool2d(input: Tensor, kernel: [number, number]): Tensor {
    const [batch, channel] = input.shape;
    const [tiled, newHeight, newWidth] = tile(input, kernel);
    return tiled.max(4).view(batch!, channel!, newHeight, newWidth);
}

export function softmax(input: Tensor, dim: number): Tensor {
    const m = input.max(dim);
    const e = input.sub(m).exp();
    return e.mul(e.sum(dim).inv());
}

export function logsoftmax(input: Tensor, dim: number): Tensor {
    const m = input.max(dim);
    const shifted = input.sub(m);
    const logSumExp = shifted.exp().sum(dim).log();
    return shifted.sub(logSumExp);
}

export function dropout(input: Tensor, rate: number = 0.5, ignore: boolean = false): Tensor {
    if (ignore || rate === 0.0) return input;
    if (rate >= 1.0) return Tensor.zeros(input.shape);
    const mask = Tensor.rand(input.shape).gt(rate);
    return input.mul(mask).mul(1 / (1 - rate));
}