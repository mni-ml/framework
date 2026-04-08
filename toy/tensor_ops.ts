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