import type {
    Storage,
    Shape,
    Strides,
} from './tensor_data.js';

import {
    TensorData,
    indexToPosition,
    toIndex,
    shapeProduct,
    strides as computeStrides,
    broadcastIndex,
    createSharedStorage,
    shapeBroadcast,
} from './tensor_data.js';

import { Tensor } from './tensor.js';

type GpuMatMulFn = typeof import('./gpu_ops.js')['gpuTensorMatrixMultiply'];
let _gpuMatMul: GpuMatMulFn | null | false = null;
let _gpuCallFailed = false;

async function getGpuMatMul(): Promise<GpuMatMulFn | null> {
    if (_gpuCallFailed || _gpuMatMul === false) return null;
    if (_gpuMatMul !== null) return _gpuMatMul;
    try {
        const mod = await import('./gpu_ops.js');
        _gpuMatMul = mod.gpuTensorMatrixMultiply;
        return _gpuMatMul;
    } catch {
        _gpuMatMul = false;
        return null;
    }
}

export function tensorMap(
    fn: (x: number) => number
): (
    outStorage: Storage,
    outShape: Shape,
    outStrides: Strides,
    inStorage: Storage,
    inShape: Shape,
    inStrides: Strides
) => void {
    return (
        outStorage: Storage,
        outShape: Shape,
        outStrides: Strides,
        inStorage: Storage,
        inShape: Shape,
        inStrides: Strides
    ): void => {
        const size = shapeProduct(outShape);
        const outIndex: number[] = new Array(outShape.length).fill(0);
        const inIndex: number[] = new Array(inShape.length).fill(0);

        for (let ordinal = 0; ordinal < size; ordinal++) {
            toIndex(ordinal, outShape, outIndex);

            broadcastIndex(outIndex, outShape, inShape, inIndex);

            const inPos = indexToPosition(inIndex, inStrides);
            const outPos = indexToPosition(outIndex, outStrides);
            outStorage[outPos] = fn(inStorage[inPos]!);
        }
    };
}

export function tensorZip(
    fn: (a: number, b: number) => number
): (
    outStorage: Storage,
    outShape: Shape,
    outStrides: Strides,
    aStorage: Storage,
    aShape: Shape,
    aStrides: Strides,
    bStorage: Storage,
    bShape: Shape,
    bStrides: Strides
) => void {
    return (
        outStorage: Storage,
        outShape: Shape,
        outStrides: Strides,
        aStorage: Storage,
        aShape: Shape,
        aStrides: Strides,
        bStorage: Storage,
        bShape: Shape,
        bStrides: Strides
    ): void => {
        const size = shapeProduct(outShape);
        const outIndex: number[] = new Array(outShape.length).fill(0);
        const aIndex: number[] = new Array(aShape.length).fill(0);
        const bIndex: number[] = new Array(bShape.length).fill(0);

        for (let ordinal = 0; ordinal < size; ordinal++) {
            toIndex(ordinal, outShape, outIndex);

            broadcastIndex(outIndex, outShape, aShape, aIndex);
            broadcastIndex(outIndex, outShape, bShape, bIndex);

            const aPos = indexToPosition(aIndex, aStrides);
            const bPos = indexToPosition(bIndex, bStrides);
            const outPos = indexToPosition(outIndex, outStrides);
            outStorage[outPos] = fn(aStorage[aPos]!, bStorage[bPos]!);
        }
    }
}

export function tensorReduce(
    fn: (acc: number, x: number) => number
): (
    outStorage: Storage,
    outShape: Shape,
    outStrides: Strides,
    aStorage: Storage,
    aShape: Shape,
    aStrides: Strides,
    reduceDim: number
) => void {
    return (
        outStorage: Storage,
        outShape: Shape,
        outStrides: Strides,
        aStorage: Storage,
        aShape: Shape,
        aStrides: Strides,
        reduceDim: number
    ): void => {
        const outSize = shapeProduct(outShape);
        const reduceDimSize = aShape[reduceDim]!;
        const outIndex: number[] = new Array(outShape.length).fill(0);
        const aIndex: number[] = new Array(aShape.length).fill(0);
        for (let ordinal = 0; ordinal < outSize; ordinal++) {
            toIndex(ordinal, outShape, outIndex);

            for (let i = 0; i < outShape.length; i++) {
                aIndex[i] = outIndex[i]!;
            }

            const outPos = indexToPosition(outIndex, outStrides);

            aIndex[reduceDim] = 0;
            let acc = aStorage[indexToPosition(aIndex, aStrides)]!;

            for (let j = 1; j < reduceDimSize; j++) {
                aIndex[reduceDim] = j;
                const aPos = indexToPosition(aIndex, aStrides);
                acc = fn(acc, aStorage[aPos]!);
            }

            outStorage[outPos] = acc;
        }
    }
}

function isContiguous(shape: Shape, strs: Strides): boolean {
    const natural = computeStrides(shape);
    for (let i = 0; i < shape.length; i++) {
        if (strs[i] !== natural[i]) return false;
    }
    return true;
}

function makeContiguous(data: { storage: Storage; shape: Shape; strides: Strides; size: number }): { storage: Storage; shape: Shape; strides: Strides } {
    if (isContiguous(data.shape, data.strides)) return data;
    const out = createSharedStorage(data.size);
    const idx: number[] = new Array(data.shape.length).fill(0);
    for (let i = 0; i < data.size; i++) {
        toIndex(i, data.shape, idx);
        out[i] = data.storage[indexToPosition(idx, data.strides)]!;
    }
    return { storage: out, shape: data.shape, strides: computeStrides(data.shape) };
}

/**
 * Compute batch offset for a given output batch ordinal, handling broadcast.
 * outBatchDims is the broadcast result shape, inputBatchDims is the original.
 */
function buildBatchMap(outBatchDims: number[], inputBatchDims: number[]): Int32Array {
    const outBatchSize = shapeProduct(outBatchDims);
    const map = new Int32Array(outBatchSize);
    const numDims = outBatchDims.length;
    const inputLen = inputBatchDims.length;
    const offset = numDims - inputLen;

    for (let batch = 0; batch < outBatchSize; batch++) {
        let remaining = batch;
        let inputOrdinal = 0;
        let inputStride = 1;
        for (let d = inputLen - 1; d >= 0; d--) {
            inputStride = d < inputLen - 1 ? inputStride * inputBatchDims[d + 1]! : 1;
        }
        const inputStrides = new Array(inputLen);
        let s = 1;
        for (let d = inputLen - 1; d >= 0; d--) {
            inputStrides[d] = s;
            s *= inputBatchDims[d]!;
        }
        remaining = batch;
        const outIdx = new Array(numDims);
        for (let d = numDims - 1; d >= 0; d--) {
            outIdx[d] = remaining % outBatchDims[d]!;
            remaining = Math.floor(remaining / outBatchDims[d]!);
        }
        inputOrdinal = 0;
        for (let d = 0; d < inputLen; d++) {
            const idx = inputBatchDims[d] === 1 ? 0 : outIdx[d + offset]!;
            inputOrdinal += idx * inputStrides[d];
        }
        map[batch] = inputOrdinal;
    }
    return map;
}

function cpuMatMul(
    aStorage: Storage, aBatchDims: number[], M: number, K: number,
    bStorage: Storage, bBatchDims: number[], N: number,
    outBatchDims: number[],
): Storage {
    const outBatchSize = shapeProduct(outBatchDims);
    const outSize = outBatchSize * M * N;
    const outStorage = createSharedStorage(outSize);
    const aMK = M * K;
    const bKN = K * N;
    const outMN = M * N;

    const aBatchMap = buildBatchMap(outBatchDims, aBatchDims);
    const bBatchMap = buildBatchMap(outBatchDims, bBatchDims);

    for (let batch = 0; batch < outBatchSize; batch++) {
        const aOff = aBatchMap[batch]! * aMK;
        const bOff = bBatchMap[batch]! * bKN;
        const outOff = batch * outMN;

        for (let i = 0; i < M; i++) {
            const aRowBase = aOff + i * K;
            const outRowBase = outOff + i * N;
            for (let k = 0; k < K; k++) {
                const a_ik = aStorage[aRowBase + k]!;
                const bRowBase = bOff + k * N;
                for (let j = 0; j < N; j++) {
                    outStorage[outRowBase + j] = outStorage[outRowBase + j]! + a_ik * bStorage[bRowBase + j]!;
                }
            }
        }
    }
    return outStorage;
}

/**
 * Async matrix multiply: tries GPU first, falls back to optimized CPU.
 * Supports arbitrary batch dimensions with broadcasting.
 */
export async function tensorMatrixMultiply(A: Tensor, B: Tensor): Promise<Tensor> {
    const M = A.shape[A.shape.length - 2]!;
    const K = A.shape[A.shape.length - 1]!;
    const K2 = B.shape[B.shape.length - 2]!;
    const N = B.shape[B.shape.length - 1]!;

    if (!M || !K || !K2 || !N) return A;
    if (K !== K2) throw new Error("A is of shape MxK. Expected B of shape K2xN");

    const both2D = A.data.shape.length === 2 && B.data.shape.length === 2;
    const aBatchDims = A.data.shape.length <= 2 ? [1] : [...A.shape.slice(0, -2)] as number[];
    const bBatchDims = B.data.shape.length <= 2 ? [1] : [...B.shape.slice(0, -2)] as number[];
    const outBatchDims = shapeBroadcast(aBatchDims, bBatchDims) as number[];

    const aContig = makeContiguous(A.data);
    const bContig = makeContiguous(B.data);

    const aFullShape = [...aBatchDims, M, K];
    const bFullShape = [...bBatchDims, K, N];
    const outShape = [...outBatchDims, M, N];
    const outSize = shapeProduct(outShape);
    const outStrides = computeStrides(outShape);

    const aStrides = computeStrides(aFullShape);
    const bStrides = computeStrides(bFullShape);

    let outStorage: Storage;
    const gpuFn = await getGpuMatMul();

    if (gpuFn) {
        try {
            outStorage = createSharedStorage(outSize);
            await gpuFn(
                outStorage, outShape, outStrides, outSize,
                aContig.storage, aFullShape, aStrides,
                bContig.storage, bFullShape, bStrides,
            );
        } catch {
            _gpuCallFailed = true;
            outStorage = cpuMatMul(
                aContig.storage, aBatchDims, M, K,
                bContig.storage, bBatchDims, N,
                outBatchDims,
            );
        }
    } else {
        outStorage = cpuMatMul(
            aContig.storage, aBatchDims, M, K,
            bContig.storage, bBatchDims, N,
            outBatchDims,
        );
    }

    const finalShape = both2D ? [M, N] : outShape;
    return new Tensor(new TensorData(outStorage, finalShape));
}

/**
 * Low-level 1D convolution kernel operating on raw Storage/Shape/Strides.
 *
 * Input shape:  [batch, in_channels, width]
 * Weight shape: [out_channels, in_channels, kernel_width]
 * Output shape: [batch, out_channels, out_width]  (caller pre-allocates)
 *
 * When reverse=false: output[b,oc,t] = sum_{ic,k} input[b,ic,t+k] * weight[oc,ic,k]
 * When reverse=true:  output[b,oc,t] = sum_{ic,k} input[b,ic,t-k] * weight[oc,ic,k]
 *
 * Out-of-bounds input positions are treated as 0.
 */
export function _tensorConv1d(
    outStorage: Storage, outShape: Shape, outStrides: Strides,
    inputStorage: Storage, inputShape: Shape, inputStrides: Strides,
    weightStorage: Storage, weightShape: Shape, weightStrides: Strides,
    reverse: boolean,
): void {
    const outSize = shapeProduct(outShape);
    const inChannels = inputShape[1]!;
    const width = inputShape[2]!;
    const kw = weightShape[2]!;

    const outIndex = [0, 0, 0];
    const inputIndex = [0, 0, 0];
    const weightIndex = [0, 0, 0];

    for (let ordinal = 0; ordinal < outSize; ordinal++) {
        toIndex(ordinal, outShape, outIndex);
        const b = outIndex[0]!;
        const oc = outIndex[1]!;
        const t = outIndex[2]!;

        let val = 0;
        for (let ic = 0; ic < inChannels; ic++) {
            for (let k = 0; k < kw; k++) {
                const s = reverse ? t - k : t + k;
                if (s >= 0 && s < width) {
                    inputIndex[0] = b;
                    inputIndex[1] = ic;
                    inputIndex[2] = s;
                    weightIndex[0] = oc;
                    weightIndex[1] = ic;
                    weightIndex[2] = k;
                    val += inputStorage[indexToPosition(inputIndex, inputStrides)]!
                         * weightStorage[indexToPosition(weightIndex, weightStrides)]!;
                }
            }
        }

        outStorage[indexToPosition(outIndex, outStrides)] = val;
    }
}

/**
 * 1D convolution: input [batch, in_channels, width] x weight [out_channels, in_channels, kw]
 * -> output [batch, out_channels, width].
 */
export function tensorConv1d(
    input: Tensor, weight: Tensor, reverse: boolean = false,
): Tensor {
    const batch = input.shape[0]!;
    const inChannels = input.shape[1]!;
    const width = input.shape[2]!;
    const outChannels = weight.shape[0]!;
    const weightInChannels = weight.shape[1]!;

    if (inChannels !== weightInChannels) {
        throw new Error(
            `Conv1d channel mismatch: input has ${inChannels} channels but weight expects ${weightInChannels}`,
        );
    }

    const outShape: Shape = [batch, outChannels, width];
    const out = Tensor.zeros(outShape);

    _tensorConv1d(
        out.data.storage, out.data.shape, out.data.strides,
        input.data.storage, input.data.shape, input.data.strides,
        weight.data.storage, weight.data.shape, weight.data.strides,
        reverse,
    );

    return out;
}

/**
 * Low-level 2D convolution kernel operating on raw Storage/Shape/Strides.
 *
 * Input shape:  [batch, in_channels, height, width]
 * Weight shape: [out_channels, in_channels, kH, kW]
 * Output shape: [batch, out_channels, out_height, out_width]  (caller pre-allocates)
 *
 * When reverse=false: output[b,oc,h,w] = sum_{ic,kh,kw} input[b,ic,h+kh,w+kw] * weight[oc,ic,kh,kw]
 * When reverse=true:  output[b,oc,h,w] = sum_{ic,kh,kw} input[b,ic,h-kh,w-kw] * weight[oc,ic,kh,kw]
 *
 * Out-of-bounds input positions are treated as 0.
 */
export function _tensorConv2d(
    outStorage: Storage, outShape: Shape, outStrides: Strides,
    inputStorage: Storage, inputShape: Shape, inputStrides: Strides,
    weightStorage: Storage, weightShape: Shape, weightStrides: Strides,
    reverse: boolean,
): void {
    const outSize = shapeProduct(outShape);
    const inChannels = inputShape[1]!;
    const height = inputShape[2]!;
    const width = inputShape[3]!;
    const kH = weightShape[2]!;
    const kW = weightShape[3]!;

    const outIndex = [0, 0, 0, 0];
    const inputIndex = [0, 0, 0, 0];
    const weightIndex = [0, 0, 0, 0];

    for (let ordinal = 0; ordinal < outSize; ordinal++) {
        toIndex(ordinal, outShape, outIndex);
        const b = outIndex[0]!;
        const oc = outIndex[1]!;
        const h = outIndex[2]!;
        const w = outIndex[3]!;

        let val = 0;
        for (let ic = 0; ic < inChannels; ic++) {
            for (let kh = 0; kh < kH; kh++) {
                const sh = reverse ? h - kh : h + kh;
                if (sh < 0 || sh >= height) continue;
                for (let kw = 0; kw < kW; kw++) {
                    const sw = reverse ? w - kw : w + kw;
                    if (sw >= 0 && sw < width) {
                        inputIndex[0] = b;
                        inputIndex[1] = ic;
                        inputIndex[2] = sh;
                        inputIndex[3] = sw;
                        weightIndex[0] = oc;
                        weightIndex[1] = ic;
                        weightIndex[2] = kh;
                        weightIndex[3] = kw;
                        val += inputStorage[indexToPosition(inputIndex, inputStrides)]!
                             * weightStorage[indexToPosition(weightIndex, weightStrides)]!;
                    }
                }
            }
        }

        outStorage[indexToPosition(outIndex, outStrides)] = val;
    }
}

/**
 * 2D convolution: input [batch, in_channels, height, width] x weight [out_channels, in_channels, kH, kW]
 * -> output [batch, out_channels, height, width].
 */
export function tensorConv2d(
    input: Tensor, weight: Tensor, reverse: boolean = false,
): Tensor {
    const batch = input.shape[0]!;
    const inChannels = input.shape[1]!;
    const height = input.shape[2]!;
    const width = input.shape[3]!;
    const outChannels = weight.shape[0]!;
    const weightInChannels = weight.shape[1]!;

    if (inChannels !== weightInChannels) {
        throw new Error(
            `Conv2d channel mismatch: input has ${inChannels} channels but weight expects ${weightInChannels}`,
        );
    }

    const outShape: Shape = [batch, outChannels, height, width];
    const out = Tensor.zeros(outShape);

    _tensorConv2d(
        out.data.storage, out.data.shape, out.data.strides,
        input.data.storage, input.data.shape, input.data.strides,
        weight.data.storage, weight.data.shape, weight.data.strides,
        reverse,
    );

    return out;
}
