import { describe, test, expect, afterAll } from '@jest/globals';
import { _sumPractice, gpuTensorMap, gpuTensorZip, gpuTensorReduce, gpuTensorMatrixMultiply, gpuTensorMatrixMultiplyTensorCore } from './gpu_ops.js';
import { tensorMap, tensorZip, tensorReduce } from './tensor_ops.js';
import { TensorData, shapeProduct, toIndex, broadcastIndex, indexToPosition } from './tensor_data.js';
import type { Storage, Shape, Strides } from './tensor_data.js';
import { destroyDevice, getTensorCoreDevice, destroyTensorCoreDevice } from './gpu_backend.js';
import { WORKGROUP_SIZE } from './gpu_kernels.js';
import * as ops from './operators.js';

afterAll(() => {
    destroyTensorCoreDevice();
    destroyDevice();
});

// f32 tolerance: GPU uses f32 while CPU uses f64
const F32_TOL = 1e-4;

function expectClose(actual: Float64Array, expected: Float64Array | number[], tol = F32_TOL) {
    const exp = expected instanceof Float64Array ? expected : new Float64Array(expected);
    expect(actual.length).toBe(exp.length);
    for (let i = 0; i < actual.length; i++) {
        expect(actual[i]).toBeCloseTo(exp[i]!, tol > 1e-3 ? 3 : 5);
    }
}

// ============================================================
// _sumPractice
// ============================================================

describe('_sumPractice', () => {
    test('sums a single block', async () => {
        const size = WORKGROUP_SIZE;
        const a = new Float64Array(size);
        for (let i = 0; i < size; i++) a[i] = 1;
        const out = new Float64Array(1);

        await _sumPractice(out, a, size);
        expect(out[0]).toBeCloseTo(size, 1);
    });

    test('sums multiple full blocks', async () => {
        const numBlocks = 4;
        const size = numBlocks * WORKGROUP_SIZE;
        const a = new Float64Array(size);
        for (let i = 0; i < size; i++) a[i] = 1;
        const out = new Float64Array(numBlocks);

        await _sumPractice(out, a, size);
        for (let i = 0; i < numBlocks; i++) {
            expect(out[i]).toBeCloseTo(WORKGROUP_SIZE, 1);
        }
    });

    test('handles partial last block', async () => {
        const size = WORKGROUP_SIZE + 10;
        const a = new Float64Array(size);
        for (let i = 0; i < size; i++) a[i] = 2;
        const numBlocks = Math.ceil(size / WORKGROUP_SIZE);
        const out = new Float64Array(numBlocks);

        await _sumPractice(out, a, size);
        expect(out[0]).toBeCloseTo(WORKGROUP_SIZE * 2, 1);
        expect(out[1]).toBeCloseTo(10 * 2, 1);
    });

    test('sums range of values', async () => {
        const size = WORKGROUP_SIZE;
        const a = new Float64Array(size);
        let expectedSum = 0;
        for (let i = 0; i < size; i++) {
            a[i] = i + 1;
            expectedSum += i + 1;
        }
        const out = new Float64Array(1);

        await _sumPractice(out, a, size);
        expect(out[0]).toBeCloseTo(expectedSum, 0);
    });
});

// ============================================================
// gpuTensorMap
// ============================================================

describe('gpuTensorMap', () => {
    test('identity map preserves values', async () => {
        const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const output = TensorData.zeros([2, 3]);

        const mapFn = gpuTensorMap(ops.id);
        await mapFn(output.storage, output.shape, output.strides,
                    input.storage, input.shape, input.strides);

        expectClose(output.storage, [1, 2, 3, 4, 5, 6]);
    });

    test('negation map', async () => {
        const input = new TensorData(new Float64Array([1, -2, 3, -4]), [2, 2]);
        const output = TensorData.zeros([2, 2]);

        const mapFn = gpuTensorMap(ops.neg);
        await mapFn(output.storage, output.shape, output.strides,
                    input.storage, input.shape, input.strides);

        expectClose(output.storage, [-1, 2, -3, 4]);
    });

    test('relu map', async () => {
        const input = new TensorData(new Float64Array([-2, -1, 0, 1, 2, 3]), [6]);
        const output = TensorData.zeros([6]);

        const mapFn = gpuTensorMap(ops.relu);
        await mapFn(output.storage, output.shape, output.strides,
                    input.storage, input.shape, input.strides);

        expectClose(output.storage, [0, 0, 0, 1, 2, 3]);
    });

    test('exp map', async () => {
        const input = new TensorData(new Float64Array([0, 1, 2]), [3]);
        const output = TensorData.zeros([3]);

        const mapFn = gpuTensorMap(ops.exp);
        await mapFn(output.storage, output.shape, output.strides,
                    input.storage, input.shape, input.strides);

        expectClose(output.storage, [1, Math.exp(1), Math.exp(2)]);
    });

    test('sigmoid map', async () => {
        const input = new TensorData(new Float64Array([0, 2, -2]), [3]);
        const output = TensorData.zeros([3]);

        const mapFn = gpuTensorMap(ops.sigmoid);
        await mapFn(output.storage, output.shape, output.strides,
                    input.storage, input.shape, input.strides);

        const expected = [0.5, 1 / (1 + Math.exp(-2)), Math.exp(-2) / (1 + Math.exp(-2))];
        expectClose(output.storage, expected);
    });

    test('log map', async () => {
        const input = new TensorData(new Float64Array([1, Math.E, 10]), [3]);
        const output = TensorData.zeros([3]);

        const mapFn = gpuTensorMap(ops.log);
        await mapFn(output.storage, output.shape, output.strides,
                    input.storage, input.shape, input.strides);

        expectClose(output.storage, [0, 1, Math.log(10)]);
    });

    test('inv map', async () => {
        const input = new TensorData(new Float64Array([1, 2, 4, 5]), [4]);
        const output = TensorData.zeros([4]);

        const mapFn = gpuTensorMap(ops.inv);
        await mapFn(output.storage, output.shape, output.strides,
                    input.storage, input.shape, input.strides);

        expectClose(output.storage, [1, 0.5, 0.25, 0.2]);
    });

    test('map with broadcasting - 1D to 2D', async () => {
        const input = new TensorData(new Float64Array([1, 2, 3]), [3]);
        const output = TensorData.zeros([2, 3]);

        const mapFn = gpuTensorMap(ops.id);
        await mapFn(output.storage, output.shape, output.strides,
                    input.storage, input.shape, input.strides);

        expectClose(output.storage, [1, 2, 3, 1, 2, 3]);
    });

    test('map with broadcasting - column vector to 2D', async () => {
        const input = new TensorData(new Float64Array([10, 20]), [2, 1]);
        const output = TensorData.zeros([2, 3]);

        const mapFn = gpuTensorMap(ops.id);
        await mapFn(output.storage, output.shape, output.strides,
                    input.storage, input.shape, input.strides);

        expectClose(output.storage, [10, 10, 10, 20, 20, 20]);
    });

    test('map on non-contiguous tensor (permuted)', async () => {
        const original = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const permuted = original.permute(1, 0);
        const output = TensorData.zeros([3, 2]);

        const mapFn = gpuTensorMap(ops.neg);
        await mapFn(output.storage, output.shape, output.strides,
                    permuted.storage, permuted.shape, permuted.strides);

        expectClose(output.storage, [-1, -4, -2, -5, -3, -6]);
    });

    test('parity with CPU tensorMap for neg', async () => {
        const input = new TensorData(new Float64Array([1.5, -2.3, 0, 4.7, -0.1, 8.9]), [2, 3]);
        const cpuOut = TensorData.zeros([2, 3]);
        const gpuOut = TensorData.zeros([2, 3]);

        tensorMap(ops.neg)(
            cpuOut.storage, cpuOut.shape, cpuOut.strides,
            input.storage, input.shape, input.strides,
        );
        await gpuTensorMap(ops.neg)(
            gpuOut.storage, gpuOut.shape, gpuOut.strides,
            input.storage, input.shape, input.strides,
        );

        expectClose(gpuOut.storage, cpuOut.storage);
    });
});

// ============================================================
// gpuTensorZip
// ============================================================

describe('gpuTensorZip', () => {
    test('element-wise addition', async () => {
        const a = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
        const b = new TensorData(new Float64Array([10, 20, 30, 40]), [2, 2]);
        const output = TensorData.zeros([2, 2]);

        const zipFn = gpuTensorZip(ops.add);
        await zipFn(output.storage, output.shape, output.strides,
                    a.storage, a.shape, a.strides,
                    b.storage, b.shape, b.strides);

        expectClose(output.storage, [11, 22, 33, 44]);
    });

    test('element-wise multiplication', async () => {
        const a = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
        const b = new TensorData(new Float64Array([2, 3, 4, 5]), [2, 2]);
        const output = TensorData.zeros([2, 2]);

        const zipFn = gpuTensorZip(ops.mul);
        await zipFn(output.storage, output.shape, output.strides,
                    a.storage, a.shape, a.strides,
                    b.storage, b.shape, b.strides);

        expectClose(output.storage, [2, 6, 12, 20]);
    });

    test('element-wise max', async () => {
        const a = new TensorData(new Float64Array([1, 5, 3, 7]), [4]);
        const b = new TensorData(new Float64Array([4, 2, 6, 1]), [4]);
        const output = TensorData.zeros([4]);

        const zipFn = gpuTensorZip(ops.max);
        await zipFn(output.storage, output.shape, output.strides,
                    a.storage, a.shape, a.strides,
                    b.storage, b.shape, b.strides);

        expectClose(output.storage, [4, 5, 6, 7]);
    });

    test('element-wise lt', async () => {
        const a = new TensorData(new Float64Array([1, 5, 3, 7]), [4]);
        const b = new TensorData(new Float64Array([4, 2, 6, 7]), [4]);
        const output = TensorData.zeros([4]);

        const zipFn = gpuTensorZip(ops.lt);
        await zipFn(output.storage, output.shape, output.strides,
                    a.storage, a.shape, a.strides,
                    b.storage, b.shape, b.strides);

        expectClose(output.storage, [1, 0, 1, 0]);
    });

    test('zip with broadcasting - scalar to 2D', async () => {
        const a = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
        const b = new TensorData(new Float64Array([10]), [1]);
        const output = TensorData.zeros([2, 2]);

        const zipFn = gpuTensorZip(ops.add);
        await zipFn(output.storage, output.shape, output.strides,
                    a.storage, a.shape, a.strides,
                    b.storage, b.shape, b.strides);

        expectClose(output.storage, [11, 12, 13, 14]);
    });

    test('zip with broadcasting - row + column', async () => {
        const a = new TensorData(new Float64Array([1, 2, 3]), [1, 3]);
        const b = new TensorData(new Float64Array([10, 20]), [2, 1]);
        const output = TensorData.zeros([2, 3]);

        const zipFn = gpuTensorZip(ops.add);
        await zipFn(output.storage, output.shape, output.strides,
                    a.storage, a.shape, a.strides,
                    b.storage, b.shape, b.strides);

        expectClose(output.storage, [11, 12, 13, 21, 22, 23]);
    });

    test('parity with CPU tensorZip for mul', async () => {
        const a = new TensorData(new Float64Array([1.5, 2.3, 3.7, 0.1, 4.2, 5.6]), [2, 3]);
        const b = new TensorData(new Float64Array([0.5, 1.1, 2.0, 3.3, 0.7, 1.9]), [2, 3]);
        const cpuOut = TensorData.zeros([2, 3]);
        const gpuOut = TensorData.zeros([2, 3]);

        tensorZip(ops.mul)(
            cpuOut.storage, cpuOut.shape, cpuOut.strides,
            a.storage, a.shape, a.strides,
            b.storage, b.shape, b.strides,
        );
        await gpuTensorZip(ops.mul)(
            gpuOut.storage, gpuOut.shape, gpuOut.strides,
            a.storage, a.shape, a.strides,
            b.storage, b.shape, b.strides,
        );

        expectClose(gpuOut.storage, cpuOut.storage);
    });
});

// ============================================================
// gpuTensorReduce
// ============================================================

describe('gpuTensorReduce', () => {
    test('sum along dim 0', async () => {
        const a = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const outShape = [1, 3] as const;
        const output = TensorData.zeros(outShape);

        const reduceFn = gpuTensorReduce(ops.add);
        await reduceFn(
            output.storage, output.shape, output.strides,
            a.storage, a.shape, a.strides, 0,
        );

        expectClose(output.storage, [5, 7, 9]);
    });

    test('sum along dim 1', async () => {
        const a = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const outShape = [2, 1] as const;
        const output = TensorData.zeros(outShape);

        const reduceFn = gpuTensorReduce(ops.add);
        await reduceFn(
            output.storage, output.shape, output.strides,
            a.storage, a.shape, a.strides, 1,
        );

        expectClose(output.storage, [6, 15]);
    });

    test('mul reduce along dim 0', async () => {
        const a = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const outShape = [1, 3] as const;
        const output = TensorData.zeros(outShape);

        const reduceFn = gpuTensorReduce(ops.mul);
        await reduceFn(
            output.storage, output.shape, output.strides,
            a.storage, a.shape, a.strides, 0,
        );

        expectClose(output.storage, [4, 10, 18]);
    });

    test('max reduce along dim 1', async () => {
        const a = new TensorData(new Float64Array([3, 1, 4, 1, 5, 9]), [2, 3]);
        const outShape = [2, 1] as const;
        const output = TensorData.zeros(outShape);

        const reduceFn = gpuTensorReduce(ops.max);
        await reduceFn(
            output.storage, output.shape, output.strides,
            a.storage, a.shape, a.strides, 1,
        );

        expectClose(output.storage, [4, 9]);
    });

    test('sum reduce 1D tensor', async () => {
        const a = new TensorData(new Float64Array([1, 2, 3, 4, 5]), [5]);
        const outShape = [1] as const;
        const output = TensorData.zeros(outShape);

        const reduceFn = gpuTensorReduce(ops.add);
        await reduceFn(
            output.storage, output.shape, output.strides,
            a.storage, a.shape, a.strides, 0,
        );

        expectClose(output.storage, [15]);
    });

    test('sum reduce 3D along middle dim', async () => {
        // shape [2, 3, 2]
        const a = new TensorData(
            new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            [2, 3, 2],
        );
        const outShape = [2, 1, 2] as const;
        const output = TensorData.zeros(outShape);

        const reduceFn = gpuTensorReduce(ops.add);
        await reduceFn(
            output.storage, output.shape, output.strides,
            a.storage, a.shape, a.strides, 1,
        );

        // dim1 sums: [1+3+5, 2+4+6, 7+9+11, 8+10+12] = [9, 12, 27, 30]
        expectClose(output.storage, [9, 12, 27, 30]);
    });

    test('parity with CPU tensorReduce for sum', async () => {
        const a = new TensorData(
            new Float64Array([1.5, 2.3, 3.7, 0.1, 4.2, 5.6, 7.1, 8.4, 9.0, 0.5, 1.1, 2.0]),
            [3, 4],
        );
        const cpuOut = TensorData.zeros([3, 1]);
        const gpuOut = TensorData.zeros([3, 1]);

        tensorReduce(ops.add)(
            cpuOut.storage, cpuOut.shape, cpuOut.strides,
            a.storage, a.shape, a.strides, 1,
        );
        await gpuTensorReduce(ops.add)(
            gpuOut.storage, gpuOut.shape, gpuOut.strides,
            a.storage, a.shape, a.strides, 1,
        );

        expectClose(gpuOut.storage, cpuOut.storage);
    });
});

// ============================================================
// Large tensor correctness
// ============================================================

describe('large tensor GPU ops', () => {
    test('map neg on 10k elements', async () => {
        const size = 10000;
        const data = new Float64Array(size);
        for (let i = 0; i < size; i++) data[i] = i * 0.1;
        const input = new TensorData(data, [size]);
        const output = TensorData.zeros([size]);

        await gpuTensorMap(ops.neg)(
            output.storage, output.shape, output.strides,
            input.storage, input.shape, input.strides,
        );

        for (let i = 0; i < size; i++) {
            expect(output.storage[i]).toBeCloseTo(-i * 0.1, 2);
        }
    });

    test('zip add on 10k elements', async () => {
        const size = 10000;
        const aData = new Float64Array(size);
        const bData = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            aData[i] = i;
            bData[i] = size - i;
        }
        const a = new TensorData(aData, [size]);
        const b = new TensorData(bData, [size]);
        const output = TensorData.zeros([size]);

        await gpuTensorZip(ops.add)(
            output.storage, output.shape, output.strides,
            a.storage, a.shape, a.strides,
            b.storage, b.shape, b.strides,
        );

        for (let i = 0; i < size; i++) {
            expect(output.storage[i]).toBeCloseTo(size, 0);
        }
    });

    test('reduce sum on 10k elements', async () => {
        const size = 10000;
        const data = new Float64Array(size).fill(1);
        const a = new TensorData(data, [size]);
        const output = TensorData.zeros([1]);

        await gpuTensorReduce(ops.add)(
            output.storage, output.shape, output.strides,
            a.storage, a.shape, a.strides, 0,
        );

        expect(output.storage[0]).toBeCloseTo(size, 0);
    });
});

// ============================================================
// gpuTensorMatrixMultiply
// ============================================================

function cpuMatMul(a: TensorData, b: TensorData, out: TensorData): void {
    const aDims = a.shape.length;
    const bDims = b.shape.length;
    const outDims = out.shape.length;
    const M = a.shape[aDims - 2]!;
    const K = a.shape[aDims - 1]!;
    const N = b.shape[bDims - 1]!;

    const outIdx: number[] = new Array(outDims).fill(0);
    const aIdx: number[] = new Array(aDims).fill(0);
    const bIdx: number[] = new Array(bDims).fill(0);

    for (let ord = 0; ord < out.size; ord++) {
        toIndex(ord, out.shape, outIdx);
        const i = outIdx[outDims - 2]!;
        const j = outIdx[outDims - 1]!;

        let sum = 0;
        for (let k = 0; k < K; k++) {
            broadcastIndex(outIdx, out.shape, a.shape, aIdx);
            aIdx[aDims - 2] = i;
            aIdx[aDims - 1] = k;

            broadcastIndex(outIdx, out.shape, b.shape, bIdx);
            bIdx[bDims - 2] = k;
            bIdx[bDims - 1] = j;

            sum += a.storage[indexToPosition(aIdx, a.strides)]! *
                   b.storage[indexToPosition(bIdx, b.strides)]!;
        }

        out.storage[indexToPosition(outIdx, out.strides)] = sum;
    }
}

describe('gpuTensorMatrixMultiply', () => {
    test('simple 2x2 matmul', async () => {
        const a = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
        const b = new TensorData(new Float64Array([5, 6, 7, 8]), [2, 2]);
        const output = TensorData.zeros([2, 2]);

        await gpuTensorMatrixMultiply(
            output.storage, output.shape, output.strides, output.size,
            a.storage, a.shape, a.strides,
            b.storage, b.shape, b.strides,
        );

        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        expectClose(output.storage, [19, 22, 43, 50]);
    });

    test('non-square [2,3] x [3,4]', async () => {
        const a = new TensorData(
            new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3],
        );
        const b = new TensorData(
            new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), [3, 4],
        );
        const output = TensorData.zeros([2, 4]);

        await gpuTensorMatrixMultiply(
            output.storage, output.shape, output.strides, output.size,
            a.storage, a.shape, a.strides,
            b.storage, b.shape, b.strides,
        );

        expectClose(output.storage, [38, 44, 50, 56, 83, 98, 113, 128]);
    });

    test('batched matmul [2,3,4] x [2,4,5]', async () => {
        const aData = new Float64Array(2 * 3 * 4);
        const bData = new Float64Array(2 * 4 * 5);
        for (let i = 0; i < aData.length; i++) aData[i] = i + 1;
        for (let i = 0; i < bData.length; i++) bData[i] = (i + 1) * 0.1;
        const a = new TensorData(aData, [2, 3, 4]);
        const b = new TensorData(bData, [2, 4, 5]);
        const output = TensorData.zeros([2, 3, 5]);

        const cpuOut = TensorData.zeros([2, 3, 5]);
        cpuMatMul(a, b, cpuOut);

        await gpuTensorMatrixMultiply(
            output.storage, output.shape, output.strides, output.size,
            a.storage, a.shape, a.strides,
            b.storage, b.shape, b.strides,
        );

        // f32 accumulation across K multiplies amplifies rounding vs f64 CPU reference
        expectClose(output.storage, cpuOut.storage, 1e-2);
    });

    test('broadcast batch [1,3,4] x [2,4,5]', async () => {
        const aData = new Float64Array(1 * 3 * 4);
        const bData = new Float64Array(2 * 4 * 5);
        for (let i = 0; i < aData.length; i++) aData[i] = i + 1;
        for (let i = 0; i < bData.length; i++) bData[i] = (i + 1) * 0.5;
        const a = new TensorData(aData, [1, 3, 4]);
        const b = new TensorData(bData, [2, 4, 5]);
        const output = TensorData.zeros([2, 3, 5]);

        const cpuOut = TensorData.zeros([2, 3, 5]);
        cpuMatMul(a, b, cpuOut);

        await gpuTensorMatrixMultiply(
            output.storage, output.shape, output.strides, output.size,
            a.storage, a.shape, a.strides,
            b.storage, b.shape, b.strides,
        );

        expectClose(output.storage, cpuOut.storage);
    });

    test('large matmul [64,64] x [64,64] parity with CPU', async () => {
        const N = 64;
        const aData = new Float64Array(N * N);
        const bData = new Float64Array(N * N);
        for (let i = 0; i < N * N; i++) {
            aData[i] = Math.sin(i * 0.01);
            bData[i] = Math.cos(i * 0.01);
        }
        const a = new TensorData(aData, [N, N]);
        const b = new TensorData(bData, [N, N]);
        const gpuOut = TensorData.zeros([N, N]);
        const cpuOut = TensorData.zeros([N, N]);

        cpuMatMul(a, b, cpuOut);

        await gpuTensorMatrixMultiply(
            gpuOut.storage, gpuOut.shape, gpuOut.strides, gpuOut.size,
            a.storage, a.shape, a.strides,
            b.storage, b.shape, b.strides,
        );

        expectClose(gpuOut.storage, cpuOut.storage);
    });

    test('non-contiguous (permuted) input', async () => {
        // a is [3,2] permuted to [2,3], b is [3,4]
        const aOrig = new TensorData(
            new Float64Array([1, 2, 3, 4, 5, 6]), [3, 2],
        );
        const a = aOrig.permute(1, 0); // shape [2,3], strides [1,2]
        const b = new TensorData(
            new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), [3, 4],
        );
        const gpuOut = TensorData.zeros([2, 4]);
        const cpuOut = TensorData.zeros([2, 4]);

        cpuMatMul(a, b, cpuOut);

        await gpuTensorMatrixMultiply(
            gpuOut.storage, gpuOut.shape, gpuOut.strides, gpuOut.size,
            a.storage, a.shape, a.strides,
            b.storage, b.shape, b.strides,
        );

        expectClose(gpuOut.storage, cpuOut.storage);
    });
});

// ============================================================
// gpuTensorMatrixMultiplyTensorCore (experimental)
// ============================================================

// Precondition tests for tensor core matmul -- these never touch the GPU
// so they're safe to run alongside other tests.
describe('gpuTensorMatrixMultiplyTensorCore preconditions', () => {
    test('returns false when dims are not multiples of 8', async () => {
        const a = new TensorData(new Float64Array(2 * 3).fill(1), [2, 3]);
        const b = new TensorData(new Float64Array(3 * 4).fill(1), [3, 4]);
        const out = TensorData.zeros([2, 4]);

        const used = await gpuTensorMatrixMultiplyTensorCore(
            out.storage, out.shape, out.strides, out.size,
            a.storage, a.shape, a.strides,
            b.storage, b.shape, b.strides,
        );
        expect(used).toBe(false);
    });

    test('returns false for non-contiguous (permuted) input', async () => {
        const aOrig = new TensorData(new Float64Array(16 * 8).fill(1), [16, 8]);
        const a = aOrig.permute(1, 0); // [8, 16] but non-contiguous
        const b = new TensorData(new Float64Array(16 * 8).fill(1), [16, 8]);
        const out = TensorData.zeros([8, 8]);

        const used = await gpuTensorMatrixMultiplyTensorCore(
            out.storage, out.shape, out.strides, out.size,
            a.storage, a.shape, a.strides,
            b.storage, b.shape, b.strides,
        );
        expect(used).toBe(false);
    });
});

// Hardware tensor core test -- kept in its own describe so it can be run
// in isolation. Dawn's experimental subgroup matrix feature triggers a
// SIGSEGV during Node process teardown when combined with the regular GPU
// device; the compute results themselves are correct.
// Run with:  npx jest --testNamePattern 'tensor core matmul parity'
describe('gpuTensorMatrixMultiplyTensorCore hardware', () => {
    test('tensor core matmul parity with CPU (if hw available)', async () => {
        const tcDevice = await getTensorCoreDevice();
        if (!tcDevice) return;

        const N = 8;
        const aData = new Float64Array(N * N);
        const bData = new Float64Array(N * N);
        for (let i = 0; i < N * N; i++) {
            aData[i] = Math.sin(i * 0.1);
            bData[i] = Math.cos(i * 0.1);
        }
        const a = new TensorData(aData, [N, N]);
        const b = new TensorData(bData, [N, N]);
        const tcOut = TensorData.zeros([N, N]);
        const cpuOut = TensorData.zeros([N, N]);

        cpuMatMul(a, b, cpuOut);

        const used = await gpuTensorMatrixMultiplyTensorCore(
            tcOut.storage, tcOut.shape, tcOut.strides, tcOut.size,
            a.storage, a.shape, a.strides,
            b.storage, b.shape, b.strides,
        );
        expect(used).toBe(true);
        expectClose(tcOut.storage, cpuOut.storage, 1e-2);
    });
});
