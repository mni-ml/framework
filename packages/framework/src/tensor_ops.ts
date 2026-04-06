import type {
    Storage,
    Shape,
    Strides,
} from './tensor_data.js';

import {
    indexToPosition,
    toIndex,
    shapeProduct,
    broadcastIndex
} from './tensor_data.js';

import { Tensor } from './tensor.js';
import { shapeBroadcast } from './tensor_data.js';

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

/**
 * Matrix multiply supporting 2D and 3D inputs with batch broadcasting.
 * 2D inputs are padded to 3D internally; if both were 2D the output is squeezed back.
 */
export function tensorMatrixMultiply(A: Tensor, B: Tensor): Tensor {
    // Index from end of tensor shape, such that length - 2 is rows, length -1 is cols
    const [M, K] = [A.shape[A.shape.length - 2], A.shape[A.shape.length - 1]];
    const [K2, N] = [B.shape[B.shape.length - 2], B.shape[B.shape.length - 1]];

    if (!M || !K || !K2 || !N) {
        throw new Error("Matmul requires both inputs to be at least 2D.");
    }

    if (K !== K2) {
        throw new Error("A is of shape MxK. Expected B of shape K2xN");
    }

    let a: Tensor = A;
    let b: Tensor = B;

    // Make these always be exactly a 3 dimensional multiply, so the kernel only ever needs to deal with one batch loop + the 2D multiply
    let Ais2D = false;
    let Bis2D = false;
    if (A.data.shape.length === 2) {
        a = A.contiguous().view(1, M, K);
        Ais2D = true;
    }
    if (B.data.shape.length === 2) {
        b = B.contiguous().view(1, K2, N);
        Bis2D = true;
    }
    // If both A and B had to be converted from 2D -> 3D, then we must remove a dimension at the end. Else it will simply just disappear as per mat mult
    const both2D: boolean = Ais2D && Bis2D;

    // Get resulting dimensions as array
    const outShape = [...shapeBroadcast(a.shape.slice(0, -2), b.shape.slice(0, -2))];
    outShape.push(M);
    outShape.push(N);
    let out = Tensor.zeros(outShape);

    const size = shapeProduct(outShape);

    const outIndex: number[] = new Array(outShape.length).fill(0);
    const aIndex: number[] = new Array(a.shape.length).fill(0);
    const bIndex: number[] = new Array(b.shape.length).fill(0);

    for (let ordinal = 0; ordinal < size; ordinal++) {
        toIndex(ordinal, outShape, outIndex);
        
        broadcastIndex(outIndex, outShape, a.shape, aIndex);
        broadcastIndex(outIndex, outShape, b.shape, bIndex);

        let acc = 0;

        for (let k = 0; k < K; k++) {
            aIndex[aIndex.length - 1] = k;      // K dim in A
            bIndex[bIndex.length - 2] = k;      // K dim in B

            acc += a.get(aIndex) * b.get(bIndex);
        }

        out.set(outIndex, acc);
    }

    // Revert extra 3rd dimension
    if (both2D) {
        out = out.view(M,N);
    }

    return out;
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
