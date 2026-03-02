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
 * Parallel matrix multiply. Outer loop (output elements) in parallel
 * computes every entry of the output matrix
 * 
 * Restriction: it only handles inputs that are already 2D or 3D, and just pads 2D up to 3D
 */
export function tensorMatrixMultiply(A: Tensor, B: Tensor): Tensor {
    // Index from end of tensor shape, such that length - 2 is rows, length -1 is cols
    const [M, K] = [A.shape[A.shape.length - 2], A.shape[A.shape.length - 1]];
    const [K2, N] = [B.shape[B.shape.length - 2], B.shape[B.shape.length - 1]];

    if (!M || !K || !K2 || !N) {
        return A;
    }

    let a: Tensor = A;
    let b: Tensor = B;

    // Make these always be exactly a 3 dimensional multiply, so the kernel only ever needs to deal with one batch loop + the 2D multiply
    let Ais2D = false;
    let Bis2D = false;
    if (A.data.shape.length == 2) {
        a = A.contiguous().view(1, M, K);
        Ais2D = true;
    }
    if (B.data.shape.length == 2) {
        b = B.contiguous().view(1, K2, N);
        Bis2D = true;
    }
    // If both A and B had to be converted from 2D -> 3D, then we must remove a dimension at the end. Else it will simply just disappear as per mat mult
    const both2D: boolean = Ais2D && Bis2D;

    // Get resulting dimensions as array
    const outShape = [...shapeBroadcast(a.shape.slice(0, -2), b.shape.slice(0, -2))];
    outShape.push(M);
    outShape.push(N);

    if (K !== K2) {
        throw new Error("A is of shape MxK. Expected B of shape K2xN");
    }
    let out = Tensor.zeros(outShape);

    const size = shapeProduct(outShape);

    const outIndex: number[] = new Array(outShape.length).fill(0);
    const aIndex: number[] = new Array(a.shape.length).fill(0);
    const bIndex: number[] = new Array(b.shape.length).fill(0);

    for (let ordinal = 0; ordinal < size; ordinal++) {
        toIndex(ordinal, outShape, outIndex);
        
        broadcastIndex(outIndex, outShape, a.shape, aIndex);
        broadcastIndex(outIndex, outShape, b.shape, bIndex);
    }

    // Revert extra 3rd dimension
    if (both2D) {
        out = out.view(M,N);
    }

    return out;
}