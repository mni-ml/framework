import type { Storage, Shape, Strides } from './tensor_data.js';
import { shapeProduct } from './tensor_data.js';

import {
    getDevice,
    uploadBuffer,
    createOutputBuffer,
    readbackBuffer,
    createUniformBuffer,
    createStorageParamsBuffer,
    getOrCreatePipeline,
} from './gpu_backend.js';

import {
    WORKGROUP_SIZE,
    UNARY_OPS,
    BINARY_OPS,
    REDUCE_IDENTITY,
    resolveUnaryOp,
    resolveBinaryOp,
    buildAlignedMapShader,
    buildBroadcastMapShader,
    buildAlignedZipShader,
    buildBroadcastZipShader,
    buildSumPracticeShader,
    buildReduceShader,
} from './gpu_kernels.js';

const MAX_DIMS = 6;

// ---- Helpers ----

function shapesEqual(a: Shape, b: Shape): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

function stridesEqual(a: Strides, b: Strides): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

/**
 * Pack a u32 params struct into an ArrayBuffer for a uniform buffer.
 * All values are packed as u32; shape/stride arrays are zero-padded to MAX_DIMS.
 */
function packU32(...values: number[]): ArrayBuffer {
    const buf = new ArrayBuffer(values.length * 4);
    const view = new Uint32Array(buf);
    for (let i = 0; i < values.length; i++) {
        view[i] = values[i]!;
    }
    return buf;
}

function padToMax(arr: Shape | Strides): number[] {
    const result = new Array(MAX_DIMS).fill(0) as number[];
    for (let i = 0; i < arr.length; i++) {
        result[i] = arr[i]!;
    }
    return result;
}

function destroyAll(...bufs: GPUBuffer[]): void {
    for (const b of bufs) b.destroy();
}

// ---- _sumPractice ----

/**
 * Practice sum kernel: given array of length `size`, produce ceil(size / WORKGROUP_SIZE)
 * partial sums. Each workgroup sums WORKGROUP_SIZE contiguous elements using shared memory.
 */
export async function _sumPractice(
    out: Storage,
    a: Storage,
    size: number,
): Promise<void> {
    const device = await getDevice();
    const numBlocks = Math.ceil(size / WORKGROUP_SIZE);

    const shaderCode = buildSumPracticeShader();
    const pipeline = getOrCreatePipeline(device, shaderCode);

    const aBuf = uploadBuffer(device, a.subarray(0, size));
    const outBuf = createOutputBuffer(device, numBlocks);
    const paramBuf = createUniformBuffer(device, packU32(size));

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: aBuf } },
            { binding: 1, resource: { buffer: outBuf } },
            { binding: 2, resource: { buffer: paramBuf } },
        ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(numBlocks);
    pass.end();
    device.queue.submit([encoder.finish()]);

    const result = await readbackBuffer(device, outBuf, numBlocks);
    for (let i = 0; i < numBlocks; i++) {
        out[i] = result[i]!;
    }

    destroyAll(aBuf, outBuf, paramBuf);
}

// ---- gpuTensorMap ----

/**
 * GPU higher-order tensor map.  fn must be a known operator from operators.ts.
 */
export function gpuTensorMap(
    fn: (x: number) => number,
): (
    outStorage: Storage,
    outShape: Shape,
    outStrides: Strides,
    inStorage: Storage,
    inShape: Shape,
    inStrides: Strides,
) => Promise<void> {
    const opName = resolveUnaryOp(fn);
    const opBody = UNARY_OPS[opName]!;

    return async (
        outStorage, outShape, outStrides,
        inStorage, inShape, inStrides,
    ): Promise<void> => {
        const device = await getDevice();
        const size = shapeProduct(outShape);
        const aligned = shapesEqual(outShape, inShape) && stridesEqual(outStrides, inStrides);

        let shaderCode: string;
        let paramBuf: GPUBuffer;

        if (aligned) {
            shaderCode = buildAlignedMapShader(opBody);
            paramBuf = createUniformBuffer(device, packU32(size));
        } else {
            shaderCode = buildBroadcastMapShader(opBody);
            paramBuf = createStorageParamsBuffer(device, packU32(
                size,
                outShape.length,
                inShape.length,
                0, // padding
                ...padToMax(outShape),
                ...padToMax(outStrides),
                ...padToMax(inShape),
                ...padToMax(inStrides),
            ));
        }

        const pipeline = getOrCreatePipeline(device, shaderCode);
        const inBuf = uploadBuffer(device, inStorage);
        const outBuf = createOutputBuffer(device, size);

        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: inBuf } },
                { binding: 1, resource: { buffer: outBuf } },
                { binding: 2, resource: { buffer: paramBuf } },
            ],
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(size / WORKGROUP_SIZE));
        pass.end();
        device.queue.submit([encoder.finish()]);

        const result = await readbackBuffer(device, outBuf, size);
        for (let i = 0; i < size; i++) {
            outStorage[i] = result[i]!;
        }

        destroyAll(inBuf, outBuf, paramBuf);
    };
}

// ---- gpuTensorZip ----

/**
 * GPU higher-order tensor zip (binary map). fn must be a known binary operator.
 */
export function gpuTensorZip(
    fn: (a: number, b: number) => number,
): (
    outStorage: Storage,
    outShape: Shape,
    outStrides: Strides,
    aStorage: Storage,
    aShape: Shape,
    aStrides: Strides,
    bStorage: Storage,
    bShape: Shape,
    bStrides: Strides,
) => Promise<void> {
    const opName = resolveBinaryOp(fn);
    const opBody = BINARY_OPS[opName]!;

    return async (
        outStorage, outShape, outStrides,
        aStorage, aShape, aStrides,
        bStorage, bShape, bStrides,
    ): Promise<void> => {
        const device = await getDevice();
        const size = shapeProduct(outShape);

        const aligned =
            shapesEqual(outShape, aShape) &&
            shapesEqual(outShape, bShape) &&
            stridesEqual(outStrides, aStrides) &&
            stridesEqual(outStrides, bStrides);

        let shaderCode: string;
        let paramBuf: GPUBuffer;

        if (aligned) {
            shaderCode = buildAlignedZipShader(opBody);
            paramBuf = createUniformBuffer(device, packU32(size));
        } else {
            shaderCode = buildBroadcastZipShader(opBody);
            paramBuf = createStorageParamsBuffer(device, packU32(
                size,
                outShape.length,
                aShape.length,
                bShape.length,
                ...padToMax(outShape),
                ...padToMax(outStrides),
                ...padToMax(aShape),
                ...padToMax(aStrides),
                ...padToMax(bShape),
                ...padToMax(bStrides),
            ));
        }

        const pipeline = getOrCreatePipeline(device, shaderCode);
        const aBuf = uploadBuffer(device, aStorage);
        const bBuf = uploadBuffer(device, bStorage);
        const outBuf = createOutputBuffer(device, size);

        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: aBuf } },
                { binding: 1, resource: { buffer: bBuf } },
                { binding: 2, resource: { buffer: outBuf } },
                { binding: 3, resource: { buffer: paramBuf } },
            ],
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(size / WORKGROUP_SIZE));
        pass.end();
        device.queue.submit([encoder.finish()]);

        const result = await readbackBuffer(device, outBuf, size);
        for (let i = 0; i < size; i++) {
            outStorage[i] = result[i]!;
        }

        destroyAll(aBuf, bBuf, outBuf, paramBuf);
    };
}

// ---- gpuTensorReduce ----

/**
 * GPU higher-order tensor reduce. fn must be a known binary operator
 * with an entry in REDUCE_IDENTITY.
 * One workgroup per output element; threads cooperatively reduce
 * the target dimension using shared memory tree reduction.
 */
export function gpuTensorReduce(
    fn: (acc: number, x: number) => number,
): (
    outStorage: Storage,
    outShape: Shape,
    outStrides: Strides,
    aStorage: Storage,
    aShape: Shape,
    aStrides: Strides,
    reduceDim: number,
) => Promise<void> {
    const opName = resolveBinaryOp(fn);
    const opBody = BINARY_OPS[opName]!;
    const identity = REDUCE_IDENTITY[opName];
    if (identity === undefined) {
        throw new Error(`No reduce identity for op: ${opName}`);
    }

    return async (
        outStorage, outShape, outStrides,
        aStorage, aShape, aStrides,
        reduceDim,
    ): Promise<void> => {
        const device = await getDevice();
        const outSize = shapeProduct(outShape);
        const reduceDimSize = aShape[reduceDim]!;
        const reduceStride = aStrides[reduceDim]!;

        const shaderCode = buildReduceShader(opBody, identity);
        const pipeline = getOrCreatePipeline(device, shaderCode);

        const paramBuf = createStorageParamsBuffer(device, packU32(
            outSize,
            outShape.length,
            aShape.length,
            reduceDim,
            reduceDimSize,
            reduceStride,
            0, // _pad0
            0, // _pad1
            ...padToMax(outShape),
            ...padToMax(outStrides),
            ...padToMax(aShape),
            ...padToMax(aStrides),
        ));

        const aBuf = uploadBuffer(device, aStorage);
        const outBuf = createOutputBuffer(device, outSize);

        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: aBuf } },
                { binding: 1, resource: { buffer: outBuf } },
                { binding: 2, resource: { buffer: paramBuf } },
            ],
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(outSize);
        pass.end();
        device.queue.submit([encoder.finish()]);

        const result = await readbackBuffer(device, outBuf, outSize);
        for (let i = 0; i < outSize; i++) {
            outStorage[i] = result[i]!;
        }

        destroyAll(aBuf, outBuf, paramBuf);
    };
}
