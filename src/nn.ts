import { Tensor, native } from "./tensor.js";
import { Module, Parameter } from "./module.js";

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
        const seqLen = indices[0].length;
        const flat = indices.flat();
        const id = native.embeddingForward(this.weight.value._id, flat, batch, seqLen);
        return new Tensor(id);
    }

    forwardGpu(intBufId: number, batch: number, seqLen: number): Tensor {
        const id = native.embeddingForwardGpu(this.weight.value._id, intBufId, batch, seqLen);
        return new Tensor(id);
    }
}

// ============================================================
// Functional ops
// ============================================================

export function softmax(x: Tensor, dim: number = -1): Tensor {
    return new Tensor(native.softmaxOp(x._id, dim));
}

export function gelu(x: Tensor): Tensor {
    return new Tensor(native.gelu(x._id));
}

export function dropout(x: Tensor, rate: number = 0.0, inference: boolean = false): Tensor {
    if (inference || rate === 0) return x;
    return new Tensor(native.dropoutOp(x._id, rate, true));
}

export function crossEntropyLoss(logits: Tensor, targets: number[][]): Tensor {
    const shape = logits.shape;
    const V = shape[shape.length - 1];
    const BT = logits.size / V;

    // Flatten logits to [BT, V]
    const flatLogits = logits.view(BT, V);
    const flatTargets = targets.flat();
    const id = native.crossEntropyLoss(flatLogits._id, flatTargets);
    return new Tensor(id);
}

export function crossEntropyLossGpu(logits: Tensor, intBufId: number): Tensor {
    const id = native.crossEntropyLossGpu(logits._id, intBufId);
    return new Tensor(id);
}

export function flashAttention(q: Tensor, k: Tensor, v: Tensor, scale: number, causal: boolean = true): Tensor {
    const id = native.flashAttention(q._id, k._id, v._id, scale, causal);
    return new Tensor(id);
}

export class KvCache {
    private readonly cacheId: number;

    constructor(batchSize: number, numHeads: number, headDim: number, maxSeqLen: number, quantized: boolean = true) {
        if (typeof native.kvCacheCreate !== 'function') {
            throw new Error('kv cache is not available in this native build');
        }
        const requestedQuantized = quantized && typeof native.flashAttention === 'function';
        this.cacheId = native.kvCacheCreate(batchSize, numHeads, headDim, maxSeqLen, requestedQuantized);
    }

    decodeStep(q: Tensor, k: Tensor, v: Tensor, scale: number): Tensor {
        const id = native.kvCacheDecodeStep(this.cacheId, q._id, k._id, v._id, scale);
        return new Tensor(id);
    }

    append(k: Tensor, v: Tensor): void {
        native.kvCacheAppend(this.cacheId, k._id, v._id);
    }

    length(): number {
        return Number(native.kvCacheLen(this.cacheId));
    }

    isQuantized(): boolean {
        return Boolean(native.kvCacheQuantized(this.cacheId));
    }

    reset(): void {
        native.kvCacheReset(this.cacheId);
    }

    free(): void {
        native.kvCacheFree(this.cacheId);
    }
}

export function residualLayerNorm(x: Tensor, residual: Tensor, gamma: Tensor, beta: Tensor, eps: number = 1e-5): Tensor {
    return new Tensor(native.residualLayernorm(x._id, residual._id, gamma._id, beta._id, eps));
}

export function biasGelu(x: Tensor, bias: Tensor): Tensor {
    return new Tensor(native.biasGelu(x._id, bias._id));
}

export function layerNorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: number = 1e-5): Tensor {
    return new Tensor(native.layernormOp(x._id, gamma._id, beta._id, eps));
}

export function mseLoss(pred: Tensor, target: Tensor): Tensor {
    const diff = pred.sub(target);
    const sq = diff.mul(diff);
    return new Tensor(native.meanOp(new Tensor(native.meanOp(sq._id, -1))._id, -1));
}

export function logsoftmax(x: Tensor, dim: number = -1): Tensor {
    const sm = softmax(x, dim);
    return sm.log();
}

// ============================================================
// Conv modules
// ============================================================

export class Conv1d extends Module {
    weight!: Parameter<Tensor>;
    bias!: Parameter<Tensor>;
    inChannels: number;
    outChannels: number;
    kernelSize: number;
    stride: number;
    padding: number;

    constructor(inChannels: number, outChannels: number, kernelSize: number, stride: number = 1, padding: number = 0) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        const bound = 1 / Math.sqrt(inChannels * kernelSize);
        this.weight = new Parameter(randRange([outChannels, inChannels, kernelSize], -bound, bound));
        this.bias = new Parameter(randRange([outChannels], -bound, bound));
    }

    forward(input: Tensor): Tensor {
        const out = input.conv1d(this.weight.value, this.stride, this.padding);
        const [n, c, l] = out.shape;
        const biasExpanded = this.bias.value.view(1, c, 1);
        return out.add(biasExpanded);
    }
}

export class Conv2d extends Module {
    weight!: Parameter<Tensor>;
    bias!: Parameter<Tensor>;
    inChannels: number;
    outChannels: number;
    kernelSize: number;
    stride: number;
    padding: number;

    constructor(inChannels: number, outChannels: number, kernelSize: number, stride: number = 1, padding: number = 0) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        const bound = 1 / Math.sqrt(inChannels * kernelSize * kernelSize);
        this.weight = new Parameter(randRange([outChannels, inChannels, kernelSize, kernelSize], -bound, bound));
        this.bias = new Parameter(randRange([outChannels], -bound, bound));
    }

    forward(input: Tensor): Tensor {
        const out = input.conv2d(this.weight.value, this.stride, this.padding);
        const [n, c, h, w] = out.shape;
        const biasExpanded = this.bias.value.view(1, c, 1, 1);
        return out.add(biasExpanded);
    }
}

// ============================================================
// Utility
// ============================================================

export function randRange(shape: number[], min: number, max: number): Tensor {
    const r = Tensor.rand(shape);
    const data = r.toFloat32();
    const range = max - min;
    const scaled = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
        scaled[i] = data[i] * range + min;
    }
    const t = Tensor.fromFloat32(scaled, shape);
    t.setRequiresGrad(true);
    return t;
}

export function tile(x: Tensor, reps: number[]): Tensor {
    return new Tensor(native.tile(x._id, reps));
}

export function avgpool2d(x: Tensor, kh: number, kw: number): Tensor {
    return new Tensor(native.avgpool2D(x._id, kh, kw));
}

export function maxpool2d(x: Tensor, kh: number, kw: number): Tensor {
    return new Tensor(native.maxpool2D(x._id, kh, kw));
}

export { Module, Parameter };
