import { test, fc } from '@fast-check/jest';
import { describe, expect, afterAll } from '@jest/globals';
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { fastTensorMap, fastTensorZip, fastTensorReduce, destroyPool } from './fast_ops.js';
import { tensorMap, tensorZip, tensorReduce, _tensorConv1d, tensorConv1d, _tensorConv2d, tensorConv2d } from './tensor_ops.js';
import { TensorData, shapeProduct } from './tensor_data.js';
import { Tensor } from './tensor.js';

// Tear down worker pool so Jest can exit cleanly
afterAll(() => {
    destroyPool();
});

// ============================================================
// Arbitraries for property-based testing
// ============================================================

const smallDim = fc.integer({ min: 1, max: 5 });
const shape2D = fc.tuple(smallDim, smallDim);

// ============================================================
// Task 3.1 - fastTensorMap
// ============================================================

describe("fastTensorMap", () => {
    test("identity map preserves values", () => {
        const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const output = TensorData.zeros([2, 3]);

        const mapFn = fastTensorMap((x) => x);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);

        expect(Array.from(output.storage)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    test("negation map", () => {
        const input = new TensorData(new Float64Array([1, -2, 3, -4]), [2, 2]);
        const output = TensorData.zeros([2, 2]);

        const mapFn = fastTensorMap((x) => -x);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);

        expect(Array.from(output.storage)).toEqual([-1, 2, -3, 4]);
    });

    test("square map", () => {
        const input = new TensorData(new Float64Array([1, 2, 3, 4]), [4]);
        const output = TensorData.zeros([4]);

        const mapFn = fastTensorMap((x) => x * x);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);

        expect(Array.from(output.storage)).toEqual([1, 4, 9, 16]);
    });

    test("map with broadcasting - scalar to 2D", () => {
        const input = new TensorData(new Float64Array([5]), []);
        const output = TensorData.zeros([2, 2]);

        const mapFn = fastTensorMap((x) => x * 2);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);

        expect(Array.from(output.storage)).toEqual([10, 10, 10, 10]);
    });

    test("map with broadcasting - 1D to 2D", () => {
        const input = new TensorData(new Float64Array([1, 2, 3]), [3]);
        const output = TensorData.zeros([2, 3]);

        const mapFn = fastTensorMap((x) => x);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);

        expect(Array.from(output.storage)).toEqual([1, 2, 3, 1, 2, 3]);
    });

    test("map with broadcasting - column vector to 2D", () => {
        const input = new TensorData(new Float64Array([10, 20]), [2, 1]);
        const output = TensorData.zeros([2, 3]);

        const mapFn = fastTensorMap((x) => x);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);

        expect(Array.from(output.storage)).toEqual([10, 10, 10, 20, 20, 20]);
    });

    test.prop([shape2D])("map preserves shape", (shape) => {
        const size = shapeProduct(shape);
        const input = new TensorData(new Float64Array(size).fill(1), shape);
        const output = TensorData.zeros(shape);

        const mapFn = fastTensorMap((x) => x + 1);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);

        expect(output.shape).toEqual(shape);
        expect(output.size).toBe(size);
    });

    test("map on non-contiguous tensor (permuted)", () => {
        const original = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const permuted = original.permute(1, 0);
        const output = TensorData.zeros([3, 2]);

        const mapFn = fastTensorMap((x) => x * 10);
        mapFn(output.storage, output.shape, output.strides,
              permuted.storage, permuted.shape, permuted.strides);

        expect(Array.from(output.storage)).toEqual([10, 40, 20, 50, 30, 60]);
    });
});

// ============================================================
// Task 3.1 - fastTensorMap edge cases
// ============================================================

describe("fastTensorMap edge cases", () => {
    test("map on 0-dimensional tensor (scalar)", () => {
        const input = new TensorData(new Float64Array([5]), []);
        const output = TensorData.zeros([]);

        const mapFn = fastTensorMap((x) => x * 2);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);

        expect(output.storage[0]).toBe(10);
    });

    test("map with same input and output shape", () => {
        const input = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
        const output = TensorData.zeros([2, 2]);

        const mapFn = fastTensorMap((x) => x + 1);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);

        expect(Array.from(output.storage)).toEqual([2, 3, 4, 5]);
    });

    test("map on 4D tensor", () => {
        const input = new TensorData(new Float64Array(16).fill(1), [2, 2, 2, 2]);
        const output = TensorData.zeros([2, 2, 2, 2]);

        const mapFn = fastTensorMap((x) => x * 3);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);

        expect(output.storage.every(v => v === 3)).toBe(true);
    });
});

// ============================================================
// Task 3.1 - fastTensorZip
// ============================================================

describe("fastTensorZip", () => {
    test("element-wise addition", () => {
        const a = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
        const b = new TensorData(new Float64Array([10, 20, 30, 40]), [2, 2]);
        const output = TensorData.zeros([2, 2]);

        const zipFn = fastTensorZip((x, y) => x + y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);

        expect(Array.from(output.storage)).toEqual([11, 22, 33, 44]);
    });

    test("element-wise multiplication", () => {
        const a = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
        const b = new TensorData(new Float64Array([2, 3, 4, 5]), [2, 2]);
        const output = TensorData.zeros([2, 2]);

        const zipFn = fastTensorZip((x, y) => x * y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);

        expect(Array.from(output.storage)).toEqual([2, 6, 12, 20]);
    });

    test("element-wise subtraction", () => {
        const a = new TensorData(new Float64Array([10, 20, 30, 40]), [4]);
        const b = new TensorData(new Float64Array([1, 2, 3, 4]), [4]);
        const output = TensorData.zeros([4]);

        const zipFn = fastTensorZip((x, y) => x - y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);

        expect(Array.from(output.storage)).toEqual([9, 18, 27, 36]);
    });

    test("zip with broadcasting - scalar + 2D", () => {
        const a = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
        const b = new TensorData(new Float64Array([10]), []);
        const output = TensorData.zeros([2, 2]);

        const zipFn = fastTensorZip((x, y) => x + y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);

        expect(Array.from(output.storage)).toEqual([11, 12, 13, 14]);
    });

    test("zip with broadcasting - row + column", () => {
        const a = new TensorData(new Float64Array([1, 2, 3]), [1, 3]);
        const b = new TensorData(new Float64Array([10, 20]), [2, 1]);
        const output = TensorData.zeros([2, 3]);

        const zipFn = fastTensorZip((x, y) => x + y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);

        expect(Array.from(output.storage)).toEqual([11, 12, 13, 21, 22, 23]);
    });

    test("zip with broadcasting - 1D + 2D", () => {
        const a = new TensorData(new Float64Array([1, 2, 3]), [3]);
        const b = new TensorData(new Float64Array([10, 20, 30, 40, 50, 60]), [2, 3]);
        const output = TensorData.zeros([2, 3]);

        const zipFn = fastTensorZip((x, y) => x + y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);

        expect(Array.from(output.storage)).toEqual([11, 22, 33, 41, 52, 63]);
    });

    test("zip comparison - less than", () => {
        const a = new TensorData(new Float64Array([1, 5, 3, 7]), [4]);
        const b = new TensorData(new Float64Array([2, 4, 3, 8]), [4]);
        const output = TensorData.zeros([4]);

        const zipFn = fastTensorZip((x, y) => x < y ? 1 : 0);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);

        expect(Array.from(output.storage)).toEqual([1, 0, 0, 1]);
    });

    test.prop([shape2D])("zip addition is commutative", (shape) => {
        const size = shapeProduct(shape);
        const aData = new Float64Array(size);
        const bData = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            aData[i] = i;
            bData[i] = i * 2;
        }
        const a = new TensorData(aData, shape);
        const b = new TensorData(bData, shape);
        const output1 = TensorData.zeros(shape);
        const output2 = TensorData.zeros(shape);

        const zipFn = fastTensorZip((x, y) => x + y);
        zipFn(output1.storage, output1.shape, output1.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);
        zipFn(output2.storage, output2.shape, output2.strides,
              b.storage, b.shape, b.strides,
              a.storage, a.shape, a.strides);

        expect(Array.from(output1.storage)).toEqual(Array.from(output2.storage));
    });
});

// ============================================================
// Task 3.1 - fastTensorZip edge cases
// ============================================================

describe("fastTensorZip edge cases", () => {
    test("zip two scalar tensors", () => {
        const a = new TensorData(new Float64Array([3]), []);
        const b = new TensorData(new Float64Array([4]), []);
        const output = TensorData.zeros([]);

        const zipFn = fastTensorZip((x, y) => x + y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);

        expect(output.storage[0]).toBe(7);
    });

    test("zip on non-contiguous tensors", () => {
        const original = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const permuted = original.permute(1, 0);
        const b = new TensorData(new Float64Array([10, 20, 30, 40, 50, 60]), [3, 2]);
        const output = TensorData.zeros([3, 2]);

        const zipFn = fastTensorZip((x, y) => x + y);
        zipFn(output.storage, output.shape, output.strides,
              permuted.storage, permuted.shape, permuted.strides,
              b.storage, b.shape, b.strides);

        expect(Array.from(output.storage)).toEqual([11, 24, 32, 45, 53, 66]);
    });

    test("zip with 3D broadcasting", () => {
        const a = new TensorData(new Float64Array([1, 2]), [2]);
        const b = new TensorData(new Float64Array([10, 20, 30, 40, 50, 60, 70, 80]), [2, 2, 2]);
        const output = TensorData.zeros([2, 2, 2]);

        const zipFn = fastTensorZip((x, y) => x + y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);

        expect(Array.from(output.storage)).toEqual([11, 22, 31, 42, 51, 62, 71, 82]);
    });
});

// ============================================================
// Task 3.1 - fastTensorReduce
// ============================================================

describe("fastTensorReduce", () => {
    test("sum reduction along dim 0 - 2D tensor", () => {
        const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const output = TensorData.zeros([1, 3]);

        const reduceFn = fastTensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 0);

        expect(Array.from(output.storage)).toEqual([5, 7, 9]);
    });

    test("sum reduction along dim 1 - 2D tensor", () => {
        const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const output = TensorData.zeros([2, 1]);

        const reduceFn = fastTensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 1);

        expect(Array.from(output.storage)).toEqual([6, 15]);
    });

    test("product reduction along dim 0", () => {
        const input = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
        const output = TensorData.zeros([1, 2]);

        const reduceFn = fastTensorReduce((acc, x) => acc * x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 0);

        expect(Array.from(output.storage)).toEqual([3, 8]);
    });

    test("product reduction along dim 1", () => {
        const input = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
        const output = TensorData.zeros([2, 1]);

        const reduceFn = fastTensorReduce((acc, x) => acc * x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 1);

        expect(Array.from(output.storage)).toEqual([2, 12]);
    });

    test("max reduction along dim 0", () => {
        const input = new TensorData(new Float64Array([1, 5, 3, 4, 2, 6]), [2, 3]);
        const output = TensorData.zeros([1, 3]);

        const reduceFn = fastTensorReduce((acc, x) => Math.max(acc, x));
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 0);

        expect(Array.from(output.storage)).toEqual([4, 5, 6]);
    });

    test("min reduction along dim 1", () => {
        const input = new TensorData(new Float64Array([1, 5, 3, 4, 2, 6]), [2, 3]);
        const output = TensorData.zeros([2, 1]);

        const reduceFn = fastTensorReduce((acc, x) => Math.min(acc, x));
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 1);

        expect(Array.from(output.storage)).toEqual([1, 2]);
    });

    test("reduction on 1D tensor", () => {
        const input = new TensorData(new Float64Array([1, 2, 3, 4, 5]), [5]);
        const output = TensorData.zeros([1]);

        const reduceFn = fastTensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 0);

        expect(Array.from(output.storage)).toEqual([15]);
    });

    test("reduction on 3D tensor along dim 0", () => {
        const input = new TensorData(
            new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]),
            [2, 2, 2]
        );
        const output = TensorData.zeros([1, 2, 2]);

        const reduceFn = fastTensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 0);

        expect(Array.from(output.storage)).toEqual([6, 8, 10, 12]);
    });

    test("reduction on 3D tensor along dim 1", () => {
        const input = new TensorData(
            new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]),
            [2, 2, 2]
        );
        const output = TensorData.zeros([2, 1, 2]);

        const reduceFn = fastTensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 1);

        expect(Array.from(output.storage)).toEqual([4, 6, 12, 14]);
    });

    test("reduction on 3D tensor along dim 2", () => {
        const input = new TensorData(
            new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]),
            [2, 2, 2]
        );
        const output = TensorData.zeros([2, 2, 1]);

        const reduceFn = fastTensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 2);

        expect(Array.from(output.storage)).toEqual([3, 7, 11, 15]);
    });

    test("reduction with single element along dimension", () => {
        const input = new TensorData(new Float64Array([1, 2, 3, 4]), [1, 4]);
        const output = TensorData.zeros([1, 4]);

        const reduceFn = fastTensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 0);

        expect(Array.from(output.storage)).toEqual([1, 2, 3, 4]);
    });

    test.prop([
        fc.integer({ min: 2, max: 4 }),
        fc.integer({ min: 2, max: 4 }),
        fc.integer({ min: 0, max: 1 })
    ])("sum reduction equals manual sum", (dim0, dim1, reduceDim) => {
        const shape = [dim0, dim1] as const;
        const size = dim0 * dim1;
        const data = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            data[i] = i + 1;
        }
        const input = new TensorData(data, shape);

        const outShape = shape.map((s, i) => i === reduceDim ? 1 : s);
        const output = TensorData.zeros(outShape);

        const reduceFn = fastTensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, reduceDim);

        const reduceSize = shape[reduceDim];
        const otherSize = shape[1 - reduceDim];

        for (let i = 0; i < otherSize; i++) {
            let expected = 0;
            for (let j = 0; j < reduceSize; j++) {
                const idx = reduceDim === 0
                    ? [j, i]
                    : [i, j];
                expected += input.get(idx);
            }
            const outIdx = reduceDim === 0 ? [0, i] : [i, 0];
            expect(output.get(outIdx)).toBe(expected);
        }
    });
});

// ============================================================
// Task 3.1 - fastTensorReduce edge cases
// ============================================================

describe("fastTensorReduce edge cases", () => {
    test("reduce on non-contiguous tensor", () => {
        const original = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const permuted = original.permute(1, 0);
        const output = TensorData.zeros([3, 1]);

        const reduceFn = fastTensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 permuted.storage, permuted.shape, permuted.strides, 1);

        expect(Array.from(output.storage)).toEqual([5, 7, 9]);
    });

    test("reduce 4D tensor along middle dimension", () => {
        const size = 2 * 3 * 2 * 2;
        const data = new Float64Array(size);
        for (let i = 0; i < size; i++) data[i] = i + 1;
        const input = new TensorData(data, [2, 3, 2, 2]);
        const output = TensorData.zeros([2, 1, 2, 2]);

        const reduceFn = fastTensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 1);

        expect(output.get([0, 0, 0, 0])).toBe(15);
    });

    test("reduce with large dimension", () => {
        const input = new TensorData(new Float64Array(100).fill(1), [100]);
        const output = TensorData.zeros([1]);

        const reduceFn = fastTensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 0);

        expect(output.storage[0]).toBe(100);
    });
});

// ============================================================
// Task 3.1 - Stride-aligned fast path verification
// ============================================================

describe("stride-aligned fast path", () => {
    test("map: contiguous same-shape tensors use fast path", () => {
        const size = 1000;
        const input = TensorData.zeros([size]);
        for (let i = 0; i < size; i++) input.storage[i] = i;
        const output = TensorData.zeros([size]);

        const mapFn = fastTensorMap((x) => x * 2);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);

        for (let i = 0; i < size; i++) {
            expect(output.storage[i]).toBe(i * 2);
        }
    });

    test("zip: contiguous same-shape tensors use fast path", () => {
        const size = 1000;
        const a = TensorData.zeros([size]);
        const b = TensorData.zeros([size]);
        for (let i = 0; i < size; i++) {
            a.storage[i] = i;
            b.storage[i] = i * 10;
        }
        const output = TensorData.zeros([size]);

        const zipFn = fastTensorZip((x, y) => x + y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);

        for (let i = 0; i < size; i++) {
            expect(output.storage[i]).toBe(i + i * 10);
        }
    });
});

// ============================================================
// Task 3.1 - fast_ops matches simple tensor_ops
// ============================================================

describe("fast_ops matches tensor_ops (correctness)", () => {
    test("map: fast matches simple for 3D tensor", () => {
        const data = new Float64Array(24);
        for (let i = 0; i < 24; i++) data[i] = i - 12;
        const input = new TensorData(data, [2, 3, 4]);

        const outputFast = TensorData.zeros([2, 3, 4]);
        const outputSimple = TensorData.zeros([2, 3, 4]);

        const fn = (x: number) => x * x + 1;
        fastTensorMap(fn)(
            outputFast.storage, outputFast.shape, outputFast.strides,
            input.storage, input.shape, input.strides
        );
        tensorMap(fn)(
            outputSimple.storage, outputSimple.shape, outputSimple.strides,
            input.storage, input.shape, input.strides
        );

        expect(Array.from(outputFast.storage)).toEqual(Array.from(outputSimple.storage));
    });

    test("zip: fast matches simple for broadcast case", () => {
        const a = new TensorData(new Float64Array([1, 2, 3]), [1, 3]);
        const b = new TensorData(new Float64Array([10, 20]), [2, 1]);

        const outputFast = TensorData.zeros([2, 3]);
        const outputSimple = TensorData.zeros([2, 3]);

        const fn = (x: number, y: number) => x * y;
        fastTensorZip(fn)(
            outputFast.storage, outputFast.shape, outputFast.strides,
            a.storage, a.shape, a.strides,
            b.storage, b.shape, b.strides
        );
        tensorZip(fn)(
            outputSimple.storage, outputSimple.shape, outputSimple.strides,
            a.storage, a.shape, a.strides,
            b.storage, b.shape, b.strides
        );

        expect(Array.from(outputFast.storage)).toEqual(Array.from(outputSimple.storage));
    });

    test("reduce: fast matches simple for 3D reduction", () => {
        const data = new Float64Array(24);
        for (let i = 0; i < 24; i++) data[i] = i + 1;
        const input = new TensorData(data, [2, 3, 4]);

        const outputFast = TensorData.zeros([2, 1, 4]);
        const outputSimple = TensorData.zeros([2, 1, 4]);

        const fn = (acc: number, x: number) => acc + x;
        fastTensorReduce(fn)(
            outputFast.storage, outputFast.shape, outputFast.strides,
            input.storage, input.shape, input.strides, 1
        );
        tensorReduce(fn)(
            outputSimple.storage, outputSimple.shape, outputSimple.strides,
            input.storage, input.shape, input.strides, 1
        );

        expect(Array.from(outputFast.storage)).toEqual(Array.from(outputSimple.storage));
    });
});

// ============================================================
// Task 3.1 - Large tensor parallel correctness
// ============================================================

describe("large tensor parallel execution", () => {
    test("map: large tensor produces correct results", () => {
        const size = 10000;
        const input = TensorData.zeros([size]);
        for (let i = 0; i < size; i++) input.storage[i] = i;
        const output = TensorData.zeros([size]);

        const mapFn = fastTensorMap((x) => x * 2 + 1);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);

        for (let i = 0; i < size; i++) {
            expect(output.storage[i]).toBe(i * 2 + 1);
        }
    });

    test("zip: large tensor produces correct results", () => {
        const size = 10000;
        const a = TensorData.zeros([size]);
        const b = TensorData.zeros([size]);
        for (let i = 0; i < size; i++) {
            a.storage[i] = i;
            b.storage[i] = size - i;
        }
        const output = TensorData.zeros([size]);

        const zipFn = fastTensorZip((x, y) => x + y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);

        for (let i = 0; i < size; i++) {
            expect(output.storage[i]).toBe(size);
        }
    });

    test("reduce: large tensor produces correct results", () => {
        const rows = 100;
        const cols = 100;
        const input = TensorData.zeros([rows, cols]);
        for (let i = 0; i < rows * cols; i++) input.storage[i] = 1;
        const output = TensorData.zeros([rows, 1]);

        const reduceFn = fastTensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 1);

        for (let i = 0; i < rows; i++) {
            expect(output.storage[i]).toBe(cols);
        }
    });
});

// ============================================================
// Task 3.1 - Dispatch parity across PARALLEL_THRESHOLD
// ============================================================

describe("dispatch parity across PARALLEL_THRESHOLD", () => {
    test("training weights match below/above threshold", () => {
        const here = dirname(fileURLToPath(import.meta.url));
        const packageRoot = join(here, '..');
        const demoRoot = join(packageRoot, '..', 'demo');
        const scriptPath = join(demoRoot, 'src', 'dispatch_parity.js');
        const tscPath = join(packageRoot, 'node_modules', 'typescript', 'bin', 'tsc');
        const tsconfigPath = join(packageRoot, 'tsconfig.json');

        const env = { ...process.env };
        delete env['JEST_WORKER_ID'];
        delete env['TSTORCH_DISABLE_PARALLEL'];

        if (!existsSync(tscPath)) {
            throw new Error(
                'typescript is required to build @mni-ml/framework before running dispatch parity',
            );
        }

        const buildResult = spawnSync(
            process.execPath,
            [tscPath, '-p', tsconfigPath],
            {
                cwd: packageRoot,
                env,
                encoding: 'utf8',
                timeout: 60000,
            },
        );

        if (buildResult.error) {
            throw buildResult.error;
        }
        if (buildResult.status !== 0) {
            const stderr = buildResult.stderr?.trim();
            throw new Error(
                `framework build failed${stderr ? `: ${stderr}` : ''}`,
            );
        }

        // Run outside Jest so worker threads can use SharedArrayBuffer.
        const result = spawnSync(
            process.execPath,
            ['--no-warnings', scriptPath],
            {
                cwd: demoRoot,
                env,
                encoding: 'utf8',
                timeout: 30000,
            },
        );

        if (result.error) {
            throw result.error;
        }
        if (result.status !== 0) {
            const stderr = result.stderr?.trim();
            throw new Error(
                `dispatch parity runner failed${stderr ? `: ${stderr}` : ''}`,
            );
        }

        const output = result.stdout.trim();
        expect(output).not.toBe('');

        const { weightsBelow, weightsAbove } = JSON.parse(output) as {
            weightsBelow: number[];
            weightsAbove: number[];
        };

        expect(weightsAbove.length).toBe(weightsBelow.length);
        for (let i = 0; i < weightsBelow.length; i++) {
            expect(weightsAbove[i]).toBeCloseTo(weightsBelow[i], 8);
        }
    });
});

// ============================================================
// Task 3.1 - Backward pass / gradient parity
//
// These tests verify that the autograd system produces correct
// gradients when the forward pass runs through fast_ops. Since
// tensor_functions.ts now imports from fast_ops, all Tensor-level
// operations (neg, sigmoid, add, mul, sum, etc.) use the fast
// kernels. We numerically verify gradients with finite differences
// to ensure the backward pass is correct end-to-end.
// ============================================================

/**
 * Numerically check gradient of a tensor function using finite differences.
 * Mirrors minitorch's grad_check utility.
 */
function tensorGradCheck(
    fn: (...tensors: Tensor[]) => Tensor,
    ...inputs: Tensor[]
): void {
    const epsilon = 1e-5;

    const output = fn(...inputs);
    const scalarOutput = output.sum();
    scalarOutput.backward();

    for (let inputIdx = 0; inputIdx < inputs.length; inputIdx++) {
        const input = inputs[inputIdx]!;
        const grad = input.grad;
        expect(grad).not.toBeNull();
        expect(grad!.shape).toEqual(input.shape);

        for (let i = 0; i < input.size; i++) {
            const idx: number[] = [];
            let remaining = i;
            for (let d = input.dims - 1; d >= 0; d--) {
                idx.unshift(remaining % input.shape[d]!);
                remaining = Math.floor(remaining / input.shape[d]!);
            }

            const originalVal = input.get(idx);

            input.set(idx, originalVal + epsilon);
            const plusOutput = fn(...inputs).sum().item();

            input.set(idx, originalVal - epsilon);
            const minusOutput = fn(...inputs).sum().item();

            input.set(idx, originalVal);

            const numericalGrad = (plusOutput - minusOutput) / (2 * epsilon);
            const analyticalGrad = grad!.get(idx);

            expect(analyticalGrad).toBeCloseTo(numericalGrad, 3);
        }

        input.zero_grad_();
    }
}

describe("backward pass through fast_ops", () => {
    test("neg backward", () => {
        const a = Tensor.rand([3, 4]);
        tensorGradCheck((x) => x.neg(), a);
    });

    test("sigmoid backward", () => {
        const a = Tensor.rand([3, 4]);
        tensorGradCheck((x) => x.sigmoid(), a);
    });

    test("relu backward", () => {
        const a = Tensor.tensor([[0.5, -0.3, 0.8], [1.2, -0.1, 0.4]]);
        tensorGradCheck((x) => x.relu(), a);
    });

    test("exp backward", () => {
        const a = Tensor.rand([2, 3]);
        tensorGradCheck((x) => x.exp(), a);
    });

    test("log backward", () => {
        const size = 12;
        const storage = new Float64Array(size);
        for (let i = 0; i < size; i++) storage[i] = Math.random() * 2 + 0.5;
        const a = new Tensor(new TensorData(storage, [3, 4]));
        tensorGradCheck((x) => x.log(), a);
    });

    test("add backward", () => {
        const a = Tensor.rand([3, 4]);
        const b = Tensor.rand([3, 4]);
        tensorGradCheck((x, y) => x.add(y), a, b);
    });

    test("mul backward", () => {
        const a = Tensor.rand([3, 4]);
        const b = Tensor.rand([3, 4]);
        tensorGradCheck((x, y) => x.mul(y), a, b);
    });

    test("sum backward", () => {
        const a = Tensor.rand([3, 4]);
        tensorGradCheck((x) => x.sum(0), a);
    });

    test("sum backward along dim 1", () => {
        const a = Tensor.rand([3, 4]);
        tensorGradCheck((x) => x.sum(1), a);
    });

    test("composite forward + backward: a * sigmoid(b) + a", () => {
        const a = Tensor.rand([3, 4]);
        const b = Tensor.rand([3, 4]);
        tensorGradCheck((x, y) => x.mul(y.sigmoid()).add(x), a, b);
    });

    test("composite forward + backward: relu(a * b).sum()", () => {
        const a = Tensor.tensor([[0.5, -0.3, 0.8], [1.2, -0.1, 0.4]]);
        const b = Tensor.tensor([[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]]);
        tensorGradCheck((x, y) => x.mul(y).relu(), a, b);
    });

    test("broadcast backward: [3,1] + [1,4] -> [3,4]", () => {
        const a = Tensor.rand([3, 1]);
        const b = Tensor.rand([1, 4]);
        tensorGradCheck((x, y) => x.add(y), a, b);
    });

    test("broadcast backward: [4] * [3,4] -> [3,4]", () => {
        const a = Tensor.rand([4]);
        const b = Tensor.rand([3, 4]);
        tensorGradCheck((x, y) => x.mul(y), a, b);
    });

    test("chained reductions: sum(dim=1) then sum(dim=0)", () => {
        const a = Tensor.rand([4, 5]);
        tensorGradCheck((x) => x.sum(1).sum(0), a);
    });

    test("permute backward", () => {
        const a = Tensor.rand([2, 3, 4]);
        tensorGradCheck((x) => x.permute(2, 0, 1), a);
    });

    test("deep chain: sigmoid(relu(a * b + a).sum(1)).exp()", () => {
        const a = Tensor.rand([3, 4]);
        const b = Tensor.rand([3, 4]);
        tensorGradCheck(
            (x, y) => x.mul(y).add(x).relu().sum(1).sigmoid().exp(),
            a, b,
        );
    });
});

// ============================================================
// Task 4.1 - 1D Convolution (_tensorConv1d)
// ============================================================

describe("_tensorConv1d forward", () => {
    test("basic single-batch single-channel conv (guide example)", () => {
        // input=[1,2,3,4,5,6], weight=[5,2,3] -> [18,28,38,48,37,30]
        const input = Tensor.tensor([[[1, 2, 3, 4, 5, 6]]]);   // [1, 1, 6]
        const weight = Tensor.tensor([[[5, 2, 3]]]);            // [1, 1, 3]
        const out = Tensor.zeros([1, 1, 6]);

        _tensorConv1d(
            out.data.storage, out.data.shape, out.data.strides,
            input.data.storage, input.data.shape, input.data.strides,
            weight.data.storage, weight.data.shape, weight.data.strides,
            false,
        );

        const result = Array.from(out.data.storage);
        expect(result).toEqual([18, 28, 38, 48, 37, 30]);
    });

    test("kernel width 1 acts as channel-wise scaling", () => {
        const input = Tensor.tensor([[[1, 2, 3, 4]]]);   // [1, 1, 4]
        const weight = Tensor.tensor([[[3]]]);            // [1, 1, 1]
        const out = Tensor.zeros([1, 1, 4]);

        _tensorConv1d(
            out.data.storage, out.data.shape, out.data.strides,
            input.data.storage, input.data.shape, input.data.strides,
            weight.data.storage, weight.data.shape, weight.data.strides,
            false,
        );

        expect(Array.from(out.data.storage)).toEqual([3, 6, 9, 12]);
    });

    test("kernel width equals input width", () => {
        const input = Tensor.tensor([[[1, 2, 3]]]);    // [1, 1, 3]
        const weight = Tensor.tensor([[[1, 1, 1]]]);   // [1, 1, 3]
        const out = Tensor.zeros([1, 1, 3]);

        _tensorConv1d(
            out.data.storage, out.data.shape, out.data.strides,
            input.data.storage, input.data.shape, input.data.strides,
            weight.data.storage, weight.data.shape, weight.data.strides,
            false,
        );

        // t=0: 1+2+3=6, t=1: 2+3+0=5, t=2: 3+0+0=3
        expect(Array.from(out.data.storage)).toEqual([6, 5, 3]);
    });

    test("multi-channel: in_channels=2, out_channels=1", () => {
        // input [1, 2, 4]: two channels, width 4
        const input = Tensor.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]]);
        // weight [1, 2, 2]: one output channel, two input channels, kernel 2
        const weight = Tensor.tensor([[[1, 0], [0, 1]]]);
        const out = Tensor.zeros([1, 1, 4]);

        _tensorConv1d(
            out.data.storage, out.data.shape, out.data.strides,
            input.data.storage, input.data.shape, input.data.strides,
            weight.data.storage, weight.data.shape, weight.data.strides,
            false,
        );

        // oc=0, t=0: ic0: 1*1+2*0=1, ic1: 5*0+6*1=6 -> 7
        // oc=0, t=1: ic0: 2*1+3*0=2, ic1: 6*0+7*1=7 -> 9
        // oc=0, t=2: ic0: 3*1+4*0=3, ic1: 7*0+8*1=8 -> 11
        // oc=0, t=3: ic0: 4*1+OOB=4, ic1: 8*0+OOB=0 -> 4
        expect(Array.from(out.data.storage)).toEqual([7, 9, 11, 4]);
    });

    test("multi-output channels", () => {
        const input = Tensor.tensor([[[1, 2, 3]]]);   // [1, 1, 3]
        // weight [2, 1, 2]: two output channels
        const weight = Tensor.tensor([[[1, 0]], [[0, 1]]]);
        const out = Tensor.zeros([1, 2, 3]);

        _tensorConv1d(
            out.data.storage, out.data.shape, out.data.strides,
            input.data.storage, input.data.shape, input.data.strides,
            weight.data.storage, weight.data.shape, weight.data.strides,
            false,
        );

        // oc=0 (weight=[1,0]): [1, 2, 3]
        // oc=1 (weight=[0,1]): [2, 3, 0]
        expect(Array.from(out.data.storage)).toEqual([1, 2, 3, 2, 3, 0]);
    });

    test("batch processing (batch=2)", () => {
        const input = Tensor.tensor([
            [[1, 2, 3]],
            [[4, 5, 6]],
        ]);  // [2, 1, 3]
        const weight = Tensor.tensor([[[1, 1]]]);  // [1, 1, 2]
        const out = Tensor.zeros([2, 1, 3]);

        _tensorConv1d(
            out.data.storage, out.data.shape, out.data.strides,
            input.data.storage, input.data.shape, input.data.strides,
            weight.data.storage, weight.data.shape, weight.data.strides,
            false,
        );

        // batch 0: [1+2, 2+3, 3+0] = [3, 5, 3]
        // batch 1: [4+5, 5+6, 6+0] = [9, 11, 6]
        expect(Array.from(out.data.storage)).toEqual([3, 5, 3, 9, 11, 6]);
    });

    test("reverse convolution", () => {
        const input = Tensor.tensor([[[1, 2, 3, 4, 5, 6]]]);   // [1, 1, 6]
        const weight = Tensor.tensor([[[5, 2, 3]]]);            // [1, 1, 3]
        const out = Tensor.zeros([1, 1, 6]);

        _tensorConv1d(
            out.data.storage, out.data.shape, out.data.strides,
            input.data.storage, input.data.shape, input.data.strides,
            weight.data.storage, weight.data.shape, weight.data.strides,
            true,
        );

        // reverse: s = t - k
        // t=0: k=0->s=0: 1*5=5,  k=1->s=-1: OOB, k=2->s=-2: OOB -> 5
        // t=1: k=0->s=1: 2*5=10, k=1->s=0: 1*2=2, k=2->s=-1: OOB -> 12
        // t=2: k=0->s=2: 3*5=15, k=1->s=1: 2*2=4, k=2->s=0: 1*3=3 -> 22
        // t=3: k=0->s=3: 4*5=20, k=1->s=2: 3*2=6, k=2->s=1: 2*3=6 -> 32
        // t=4: k=0->s=4: 5*5=25, k=1->s=3: 4*2=8, k=2->s=2: 3*3=9 -> 42
        // t=5: k=0->s=5: 6*5=30, k=1->s=4: 5*2=10, k=2->s=3: 4*3=12 -> 52
        expect(Array.from(out.data.storage)).toEqual([5, 12, 22, 32, 42, 52]);
    });

    test("non-contiguous input (permuted)", () => {
        // Create a [1, 3, 1] tensor and permute to [1, 1, 3]
        const base = Tensor.tensor([[[1], [2], [3]]]);  // [1, 3, 1]
        const permuted = base.permute(0, 2, 1);         // [1, 1, 3]
        const weight = Tensor.tensor([[[1, 1]]]);        // [1, 1, 2]
        const out = Tensor.zeros([1, 1, 3]);

        _tensorConv1d(
            out.data.storage, out.data.shape, out.data.strides,
            permuted.data.storage, permuted.data.shape, permuted.data.strides,
            weight.data.storage, weight.data.shape, weight.data.strides,
            false,
        );

        // [1+2, 2+3, 3+0] = [3, 5, 3]
        expect(Array.from(out.data.storage)).toEqual([3, 5, 3]);
    });
});

// ============================================================
// Task 4.1 - tensorConv1d high-level wrapper
// ============================================================

describe("tensorConv1d wrapper", () => {
    test("matches low-level kernel", () => {
        const input = Tensor.tensor([[[1, 2, 3, 4, 5, 6]]]);
        const weight = Tensor.tensor([[[5, 2, 3]]]);

        const result = tensorConv1d(input, weight, false);

        expect(result.shape).toEqual([1, 1, 6]);
        expect(Array.from(result.data.storage)).toEqual([18, 28, 38, 48, 37, 30]);
    });

    test("channel mismatch throws", () => {
        const input = Tensor.tensor([[[1, 2, 3]]]);     // [1, 1, 3]
        const weight = Tensor.tensor([[[1, 1], [2, 2]]]); // [1, 2, 2]

        expect(() => tensorConv1d(input, weight)).toThrow(/channel mismatch/i);
    });

    test("multi-batch multi-channel", () => {
        const input = Tensor.tensor([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]);  // [2, 2, 2]
        const weight = Tensor.tensor([[[1, 0], [0, 1]]]);  // [1, 2, 2]

        const result = tensorConv1d(input, weight);

        // batch 0, oc 0: ic0: 1*1+2*0=1, ic1: 3*0+4*1=4 -> t=0: 5
        //                ic0: 2*1+OOB=2, ic1: 4*0+OOB=0 -> t=1: 2
        // batch 1, oc 0: ic0: 5*1+6*0=5, ic1: 7*0+8*1=8 -> t=0: 13
        //                ic0: 6*1+OOB=6, ic1: 8*0+OOB=0 -> t=1: 6
        expect(result.shape).toEqual([2, 1, 2]);
        expect(Array.from(result.data.storage)).toEqual([5, 2, 13, 6]);
    });
});

// ============================================================
// Task 4.1 - Tensor.conv1d method + backward
// ============================================================

describe("Tensor.conv1d", () => {
    test("forward produces correct output", () => {
        const input = Tensor.tensor([[[1, 2, 3, 4, 5, 6]]]);
        const weight = Tensor.tensor([[[5, 2, 3]]]);

        const result = input.conv1d(weight);

        expect(result.shape).toEqual([1, 1, 6]);
        expect(Array.from(result.data.storage)).toEqual([18, 28, 38, 48, 37, 30]);
    });

    test("conv1d backward - gradient check on input", () => {
        const input = Tensor.rand([1, 1, 6]);
        const weight = Tensor.rand([1, 1, 3]);
        tensorGradCheck((x, w) => x.conv1d(w), input, weight);
    });

    test("conv1d backward - gradient check multi-channel", () => {
        const input = Tensor.rand([2, 3, 5]);
        const weight = Tensor.rand([4, 3, 3]);
        tensorGradCheck((x, w) => x.conv1d(w), input, weight);
    });

    test("conv1d backward - gradient check kernel_width=1", () => {
        const input = Tensor.rand([1, 2, 4]);
        const weight = Tensor.rand([3, 2, 1]);
        tensorGradCheck((x, w) => x.conv1d(w), input, weight);
    });

    test("conv1d backward - gradient check kernel_width=width", () => {
        const input = Tensor.rand([1, 1, 4]);
        const weight = Tensor.rand([1, 1, 4]);
        tensorGradCheck((x, w) => x.conv1d(w), input, weight);
    });

    test("conv1d backward - gradient check batch=3", () => {
        const input = Tensor.rand([3, 2, 6]);
        const weight = Tensor.rand([2, 2, 3]);
        tensorGradCheck((x, w) => x.conv1d(w), input, weight);
    });

    test("conv1d chained with relu + sum", () => {
        const input = Tensor.rand([2, 2, 5]);
        const weight = Tensor.rand([3, 2, 3]);
        tensorGradCheck(
            (x, w) => x.conv1d(w).relu().sum(2),
            input, weight,
        );
    });

    test("conv1d chained with sigmoid", () => {
        const input = Tensor.rand([1, 1, 4]);
        const weight = Tensor.rand([1, 1, 2]);
        tensorGradCheck(
            (x, w) => x.conv1d(w).sigmoid(),
            input, weight,
        );
    });
});

// ============================================================
// Task 4.2 - 2D Convolution (_tensorConv2d)
// ============================================================

describe("_tensorConv2d forward", () => {
    test("basic single-batch single-channel 3x3 input, 2x2 kernel", () => {
        // input: [[1,2,3],[4,5,6],[7,8,9]]  weight: [[1,0],[0,1]]
        const input = Tensor.tensor([[[[1,2,3],[4,5,6],[7,8,9]]]]);   // [1,1,3,3]
        const weight = Tensor.tensor([[[[1,0],[0,1]]]]);              // [1,1,2,2]
        const out = Tensor.zeros([1,1,3,3]);

        _tensorConv2d(
            out.data.storage, out.data.shape, out.data.strides,
            input.data.storage, input.data.shape, input.data.strides,
            weight.data.storage, weight.data.shape, weight.data.strides,
            false,
        );

        // h=0,w=0: 1*1+2*0+4*0+5*1=6
        // h=0,w=1: 2*1+3*0+5*0+6*1=8
        // h=0,w=2: 3*1+0*0+6*0+0*1=3
        // h=1,w=0: 4*1+5*0+7*0+8*1=12
        // h=1,w=1: 5*1+6*0+8*0+9*1=14
        // h=1,w=2: 6*1+0*0+9*0+0*1=6
        // h=2,w=0: 7*1+8*0+0*0+0*1=7
        // h=2,w=1: 8*1+9*0+0*0+0*1=8
        // h=2,w=2: 9*1+0*0+0*0+0*1=9
        expect(Array.from(out.data.storage)).toEqual([6,8,3,12,14,6,7,8,9]);
    });

    test("kernel 1x1 acts as channel-wise scaling", () => {
        const input = Tensor.tensor([[[[1,2],[3,4]]]]);   // [1,1,2,2]
        const weight = Tensor.tensor([[[[3]]]]);           // [1,1,1,1]
        const out = Tensor.zeros([1,1,2,2]);

        _tensorConv2d(
            out.data.storage, out.data.shape, out.data.strides,
            input.data.storage, input.data.shape, input.data.strides,
            weight.data.storage, weight.data.shape, weight.data.strides,
            false,
        );

        expect(Array.from(out.data.storage)).toEqual([3,6,9,12]);
    });

    test("kernel size equals input size", () => {
        const input = Tensor.tensor([[[[1,2],[3,4]]]]);       // [1,1,2,2]
        const weight = Tensor.tensor([[[[1,1],[1,1]]]]);      // [1,1,2,2]
        const out = Tensor.zeros([1,1,2,2]);

        _tensorConv2d(
            out.data.storage, out.data.shape, out.data.strides,
            input.data.storage, input.data.shape, input.data.strides,
            weight.data.storage, weight.data.shape, weight.data.strides,
            false,
        );

        // h=0,w=0: 1+2+3+4=10
        // h=0,w=1: 2+0+4+0=6
        // h=1,w=0: 3+4+0+0=7
        // h=1,w=1: 4+0+0+0=4
        expect(Array.from(out.data.storage)).toEqual([10,6,7,4]);
    });

    test("multi-channel: in_channels=2, out_channels=1", () => {
        // input [1,2,2,2]
        const input = Tensor.tensor([[[[1,2],[3,4]], [[5,6],[7,8]]]]);
        // weight [1,2,1,1]
        const weight = Tensor.tensor([[[[1]], [[2]]]]);
        const out = Tensor.zeros([1,1,2,2]);

        _tensorConv2d(
            out.data.storage, out.data.shape, out.data.strides,
            input.data.storage, input.data.shape, input.data.strides,
            weight.data.storage, weight.data.shape, weight.data.strides,
            false,
        );

        // h=0,w=0: ic0: 1*1=1, ic1: 5*2=10 -> 11
        // h=0,w=1: ic0: 2*1=2, ic1: 6*2=12 -> 14
        // h=1,w=0: ic0: 3*1=3, ic1: 7*2=14 -> 17
        // h=1,w=1: ic0: 4*1=4, ic1: 8*2=16 -> 20
        expect(Array.from(out.data.storage)).toEqual([11,14,17,20]);
    });

    test("multi-output channels", () => {
        const input = Tensor.tensor([[[[1,2],[3,4]]]]);   // [1,1,2,2]
        // weight [2,1,1,1]: two output channels with 1x1 kernels
        const weight = Tensor.tensor([[[[2]]], [[[3]]]]);
        const out = Tensor.zeros([1,2,2,2]);

        _tensorConv2d(
            out.data.storage, out.data.shape, out.data.strides,
            input.data.storage, input.data.shape, input.data.strides,
            weight.data.storage, weight.data.shape, weight.data.strides,
            false,
        );

        // oc=0 (weight=2): [2,4,6,8]
        // oc=1 (weight=3): [3,6,9,12]
        expect(Array.from(out.data.storage)).toEqual([2,4,6,8,3,6,9,12]);
    });

    test("batch processing (batch=2)", () => {
        const input = Tensor.tensor([
            [[[1,2],[3,4]]],
            [[[5,6],[7,8]]],
        ]);  // [2,1,2,2]
        const weight = Tensor.tensor([[[[1,0],[0,0]]]]);  // [1,1,2,2] identity-like
        const out = Tensor.zeros([2,1,2,2]);

        _tensorConv2d(
            out.data.storage, out.data.shape, out.data.strides,
            input.data.storage, input.data.shape, input.data.strides,
            weight.data.storage, weight.data.shape, weight.data.strides,
            false,
        );

        // weight=[1,0;0,0] just picks input[h,w]*1
        // batch0: [1,2,3,4]
        // batch1: [5,6,7,8]
        expect(Array.from(out.data.storage)).toEqual([1,2,3,4,5,6,7,8]);
    });

    test("reverse convolution", () => {
        const input = Tensor.tensor([[[[1,2,3],[4,5,6],[7,8,9]]]]);  // [1,1,3,3]
        const weight = Tensor.tensor([[[[1,2],[3,4]]]]);             // [1,1,2,2]
        const out = Tensor.zeros([1,1,3,3]);

        _tensorConv2d(
            out.data.storage, out.data.shape, out.data.strides,
            input.data.storage, input.data.shape, input.data.strides,
            weight.data.storage, weight.data.shape, weight.data.strides,
            true,
        );

        // reverse: sh=h-kh, sw=w-kw
        // h=0,w=0: kh=0,kw=0->1*1=1                                          -> 1
        // h=0,w=1: kh=0,kw=0->2*1=2, kh=0,kw=1->1*2=2                       -> 4
        // h=0,w=2: kh=0,kw=0->3*1=3, kh=0,kw=1->2*2=4                       -> 7
        // h=1,w=0: kh=0,kw=0->4*1=4, kh=1,kw=0->1*3=3                       -> 7
        // h=1,w=1: kh=0,kw=0->5*1=5, kh=0,kw=1->4*2=8, kh=1,kw=0->2*3=6, kh=1,kw=1->1*4=4 -> 23
        // h=1,w=2: kh=0,kw=0->6*1=6, kh=0,kw=1->5*2=10, kh=1,kw=0->3*3=9, kh=1,kw=1->2*4=8 -> 33
        // h=2,w=0: kh=0,kw=0->7*1=7, kh=1,kw=0->4*3=12                      -> 19
        // h=2,w=1: kh=0,kw=0->8*1=8, kh=0,kw=1->7*2=14, kh=1,kw=0->5*3=15, kh=1,kw=1->4*4=16 -> 53
        // h=2,w=2: kh=0,kw=0->9*1=9, kh=0,kw=1->8*2=16, kh=1,kw=0->6*3=18, kh=1,kw=1->5*4=20 -> 63
        expect(Array.from(out.data.storage)).toEqual([1,4,7,7,23,33,19,53,63]);
    });

    test("non-contiguous input (permuted)", () => {
        // Create [1,2,2,1] and permute to [1,1,2,2]
        const base = Tensor.tensor([[[[1],[2]],[[3],[4]]]]);  // [1,2,2,1]
        const permuted = base.permute(0, 3, 1, 2);           // [1,1,2,2]
        const weight = Tensor.tensor([[[[1]]]]);              // [1,1,1,1]
        const out = Tensor.zeros([1,1,2,2]);

        _tensorConv2d(
            out.data.storage, out.data.shape, out.data.strides,
            permuted.data.storage, permuted.data.shape, permuted.data.strides,
            weight.data.storage, weight.data.shape, weight.data.strides,
            false,
        );

        expect(Array.from(out.data.storage)).toEqual([1,2,3,4]);
    });
});

// ============================================================
// Task 4.2 - tensorConv2d high-level wrapper
// ============================================================

describe("tensorConv2d wrapper", () => {
    test("matches low-level kernel", () => {
        const input = Tensor.tensor([[[[1,2,3],[4,5,6],[7,8,9]]]]);
        const weight = Tensor.tensor([[[[1,0],[0,1]]]]);

        const result = tensorConv2d(input, weight, false);

        expect(result.shape).toEqual([1,1,3,3]);
        expect(Array.from(result.data.storage)).toEqual([6,8,3,12,14,6,7,8,9]);
    });

    test("channel mismatch throws", () => {
        const input = Tensor.tensor([[[[1,2],[3,4]]]]);        // [1,1,2,2]
        const weight = Tensor.tensor([[[[1]], [[2]]]]);        // [1,2,1,1]

        expect(() => tensorConv2d(input, weight)).toThrow(/channel mismatch/i);
    });

    test("output shape is correct", () => {
        const input = Tensor.rand([2, 3, 5, 7]);
        const weight = Tensor.rand([4, 3, 3, 3]);

        const result = tensorConv2d(input, weight);

        expect(result.shape).toEqual([2, 4, 5, 7]);
    });
});

// ============================================================
// Task 4.2 - Tensor.conv2d method + backward
// ============================================================

describe("Tensor.conv2d", () => {
    test("forward produces correct output", () => {
        const input = Tensor.tensor([[[[1,2,3],[4,5,6],[7,8,9]]]]);
        const weight = Tensor.tensor([[[[1,0],[0,1]]]]);

        const result = input.conv2d(weight);

        expect(result.shape).toEqual([1,1,3,3]);
        expect(Array.from(result.data.storage)).toEqual([6,8,3,12,14,6,7,8,9]);
    });

    test("conv2d backward - gradient check single channel", () => {
        const input = Tensor.rand([1, 1, 4, 4]);
        const weight = Tensor.rand([1, 1, 2, 2]);
        tensorGradCheck((x, w) => x.conv2d(w), input, weight);
    });

    test("conv2d backward - gradient check multi-channel", () => {
        const input = Tensor.rand([2, 3, 4, 5]);
        const weight = Tensor.rand([4, 3, 3, 3]);
        tensorGradCheck((x, w) => x.conv2d(w), input, weight);
    });

    test("conv2d backward - gradient check 1x1 kernel", () => {
        const input = Tensor.rand([1, 2, 3, 3]);
        const weight = Tensor.rand([3, 2, 1, 1]);
        tensorGradCheck((x, w) => x.conv2d(w), input, weight);
    });

    test("conv2d backward - gradient check kernel = input size", () => {
        const input = Tensor.rand([1, 1, 3, 3]);
        const weight = Tensor.rand([1, 1, 3, 3]);
        tensorGradCheck((x, w) => x.conv2d(w), input, weight);
    });

    test("conv2d backward - gradient check batch=3", () => {
        const input = Tensor.rand([3, 2, 4, 4]);
        const weight = Tensor.rand([2, 2, 2, 2]);
        tensorGradCheck((x, w) => x.conv2d(w), input, weight);
    });

    test("conv2d chained with relu + sum", () => {
        const input = Tensor.rand([2, 2, 4, 4]);
        const weight = Tensor.rand([3, 2, 2, 2]);
        tensorGradCheck(
            (x, w) => x.conv2d(w).relu().sum(3),
            input, weight,
        );
    });

    test("conv2d chained with sigmoid", () => {
        const input = Tensor.rand([1, 1, 3, 3]);
        const weight = Tensor.rand([1, 1, 2, 2]);
        tensorGradCheck(
            (x, w) => x.conv2d(w).sigmoid(),
            input, weight,
        );
    });
});
