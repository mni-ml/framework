import { test, fc } from '@fast-check/jest';
import { describe, expect } from '@jest/globals';
import { TensorData, shapeProduct } from './tensor_data.js';
import * as tensorFunctions from './tensor_functions.js';

// ============================================================
// Task 2.3 - Tensor Functions Tests
// ============================================================

describe("Unary Operations", () => {
    describe("neg", () => {
        test("negates all elements", () => {
            const input = new TensorData(new Float64Array([1, -2, 3, -4]), [2, 2]);
            const output = tensorFunctions.neg(input);
            expect(Array.from(output.storage)).toEqual([-1, 2, -3, 4]);
        });

        test("negation is self-inverse", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4]), [4]);
            const output = tensorFunctions.neg(tensorFunctions.neg(input));
            expect(Array.from(output.storage)).toEqual([1, 2, 3, 4]);
        });

        test("preserves shape", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
            const output = tensorFunctions.neg(input);
            expect(output.shape).toEqual([2, 3]);
        });
    });

    describe("sigmoid", () => {
        test("sigmoid of 0 is 0.5", () => {
            const input = new TensorData(new Float64Array([0]), [1]);
            const output = tensorFunctions.sigmoid(input);
            expect(output.storage[0]).toBeCloseTo(0.5);
        });

        test("sigmoid output is in (0, 1)", () => {
            const input = new TensorData(new Float64Array([-10, -1, 0, 1, 10]), [5]);
            const output = tensorFunctions.sigmoid(input);
            for (let i = 0; i < 5; i++) {
                expect(output.storage[i]).toBeGreaterThanOrEqual(0);
                expect(output.storage[i]).toBeLessThanOrEqual(1);
            }
        });

        test("sigmoid of large positive is near 1", () => {
            const input = new TensorData(new Float64Array([10]), [1]);
            const output = tensorFunctions.sigmoid(input);
            expect(output.storage[0]).toBeGreaterThan(0.999);
        });

        test("sigmoid of large negative is near 0", () => {
            const input = new TensorData(new Float64Array([-10]), [1]);
            const output = tensorFunctions.sigmoid(input);
            expect(output.storage[0]).toBeLessThan(0.001);
        });
    });

    describe("relu", () => {
        test("relu of positive values", () => {
            const input = new TensorData(new Float64Array([1, 2, 3]), [3]);
            const output = tensorFunctions.relu(input);
            expect(Array.from(output.storage)).toEqual([1, 2, 3]);
        });

        test("relu of negative values", () => {
            const input = new TensorData(new Float64Array([-1, -2, -3]), [3]);
            const output = tensorFunctions.relu(input);
            expect(Array.from(output.storage)).toEqual([0, 0, 0]);
        });

        test("relu of mixed values", () => {
            const input = new TensorData(new Float64Array([-2, -1, 0, 1, 2]), [5]);
            const output = tensorFunctions.relu(input);
            expect(Array.from(output.storage)).toEqual([0, 0, 0, 1, 2]);
        });

        test("relu of zero is zero", () => {
            const input = new TensorData(new Float64Array([0]), [1]);
            const output = tensorFunctions.relu(input);
            expect(output.storage[0]).toBe(0);
        });
    });

    describe("log", () => {
        test("log of 1 is 0", () => {
            const input = new TensorData(new Float64Array([1]), [1]);
            const output = tensorFunctions.log(input);
            expect(output.storage[0]).toBeCloseTo(0);
        });

        test("log of e is 1", () => {
            const input = new TensorData(new Float64Array([Math.E]), [1]);
            const output = tensorFunctions.log(input);
            expect(output.storage[0]).toBeCloseTo(1);
        });

        test("log of e^2 is 2", () => {
            const input = new TensorData(new Float64Array([Math.E * Math.E]), [1]);
            const output = tensorFunctions.log(input);
            expect(output.storage[0]).toBeCloseTo(2);
        });
    });

    describe("exp", () => {
        test("exp of 0 is 1", () => {
            const input = new TensorData(new Float64Array([0]), [1]);
            const output = tensorFunctions.exp(input);
            expect(output.storage[0]).toBeCloseTo(1);
        });

        test("exp of 1 is e", () => {
            const input = new TensorData(new Float64Array([1]), [1]);
            const output = tensorFunctions.exp(input);
            expect(output.storage[0]).toBeCloseTo(Math.E);
        });

        test("exp(log(x)) = x", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4]), [4]);
            const logged = tensorFunctions.log(input);
            const output = tensorFunctions.exp(logged);
            for (let i = 0; i < 4; i++) {
                expect(output.storage[i]).toBeCloseTo(i + 1);
            }
        });
    });

    describe("inv", () => {
        test("inv of 2 is 0.5", () => {
            const input = new TensorData(new Float64Array([2]), [1]);
            const output = tensorFunctions.inv(input);
            expect(output.storage[0]).toBeCloseTo(0.5);
        });

        test("inv(inv(x)) = x", () => {
            const input = new TensorData(new Float64Array([1, 2, 4, 5]), [4]);
            const output = tensorFunctions.inv(tensorFunctions.inv(input));
            for (let i = 0; i < 4; i++) {
                expect(output.storage[i]).toBeCloseTo(input.storage[i]!);
            }
        });
    });

    describe("id (copy)", () => {
        test("id preserves values", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
            const output = tensorFunctions.id(input);
            expect(Array.from(output.storage)).toEqual([1, 2, 3, 4]);
        });

        test("id creates a new storage", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4]), [4]);
            const output = tensorFunctions.id(input);
            expect(output.storage).not.toBe(input.storage);
        });
    });
});

describe("Binary Operations", () => {
    describe("add", () => {
        test("element-wise addition", () => {
            const a = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
            const b = new TensorData(new Float64Array([10, 20, 30, 40]), [2, 2]);
            const output = tensorFunctions.add(a, b);
            expect(Array.from(output.storage)).toEqual([11, 22, 33, 44]);
        });

        test("add with scalar broadcast", () => {
            const a = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
            const b = new TensorData(new Float64Array([10]), []);
            const output = tensorFunctions.add(a, b);
            expect(Array.from(output.storage)).toEqual([11, 12, 13, 14]);
        });

        test("add with row broadcast", () => {
            const a = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
            const b = new TensorData(new Float64Array([10, 20, 30]), [3]);
            const output = tensorFunctions.add(a, b);
            expect(Array.from(output.storage)).toEqual([11, 22, 33, 14, 25, 36]);
        });

        test("addition is commutative", () => {
            const a = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
            const b = new TensorData(new Float64Array([5, 6, 7, 8]), [2, 2]);
            const ab = tensorFunctions.add(a, b);
            const ba = tensorFunctions.add(b, a);
            expect(Array.from(ab.storage)).toEqual(Array.from(ba.storage));
        });

        test("adding zero is identity", () => {
            const a = new TensorData(new Float64Array([1, 2, 3, 4]), [4]);
            const zero = new TensorData(new Float64Array([0]), []);
            const output = tensorFunctions.add(a, zero);
            expect(Array.from(output.storage)).toEqual([1, 2, 3, 4]);
        });
    });

    describe("mul", () => {
        test("element-wise multiplication", () => {
            const a = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
            const b = new TensorData(new Float64Array([2, 3, 4, 5]), [2, 2]);
            const output = tensorFunctions.mul(a, b);
            expect(Array.from(output.storage)).toEqual([2, 6, 12, 20]);
        });

        test("mul with scalar broadcast", () => {
            const a = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
            const b = new TensorData(new Float64Array([2]), []);
            const output = tensorFunctions.mul(a, b);
            expect(Array.from(output.storage)).toEqual([2, 4, 6, 8]);
        });

        test("multiplication is commutative", () => {
            const a = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
            const b = new TensorData(new Float64Array([5, 6, 7, 8]), [2, 2]);
            const ab = tensorFunctions.mul(a, b);
            const ba = tensorFunctions.mul(b, a);
            expect(Array.from(ab.storage)).toEqual(Array.from(ba.storage));
        });

        test("multiplying by one is identity", () => {
            const a = new TensorData(new Float64Array([1, 2, 3, 4]), [4]);
            const one = new TensorData(new Float64Array([1]), []);
            const output = tensorFunctions.mul(a, one);
            expect(Array.from(output.storage)).toEqual([1, 2, 3, 4]);
        });

        test("multiplying by zero gives zero", () => {
            const a = new TensorData(new Float64Array([1, 2, 3, 4]), [4]);
            const zero = new TensorData(new Float64Array([0]), []);
            const output = tensorFunctions.mul(a, zero);
            expect(Array.from(output.storage)).toEqual([0, 0, 0, 0]);
        });
    });

    describe("lt", () => {
        test("element-wise less than", () => {
            const a = new TensorData(new Float64Array([1, 5, 3, 7]), [4]);
            const b = new TensorData(new Float64Array([2, 4, 3, 8]), [4]);
            const output = tensorFunctions.lt(a, b);
            expect(Array.from(output.storage)).toEqual([1, 0, 0, 1]);
        });

        test("lt with broadcast", () => {
            const a = new TensorData(new Float64Array([1, 5, 3, 7]), [4]);
            const b = new TensorData(new Float64Array([4]), []);
            const output = tensorFunctions.lt(a, b);
            expect(Array.from(output.storage)).toEqual([1, 0, 1, 0]);
        });
    });

    describe("eq", () => {
        test("element-wise equality", () => {
            const a = new TensorData(new Float64Array([1, 2, 3, 4]), [4]);
            const b = new TensorData(new Float64Array([1, 5, 3, 8]), [4]);
            const output = tensorFunctions.eq(a, b);
            expect(Array.from(output.storage)).toEqual([1, 0, 1, 0]);
        });

        test("eq with broadcast", () => {
            const a = new TensorData(new Float64Array([1, 2, 1, 2]), [4]);
            const b = new TensorData(new Float64Array([1]), []);
            const output = tensorFunctions.eq(a, b);
            expect(Array.from(output.storage)).toEqual([1, 0, 1, 0]);
        });
    });

    describe("isClose", () => {
        test("identical values are close", () => {
            const a = new TensorData(new Float64Array([1, 2, 3, 4]), [4]);
            const b = new TensorData(new Float64Array([1, 2, 3, 4]), [4]);
            const output = tensorFunctions.isClose(a, b);
            expect(Array.from(output.storage)).toEqual([1, 1, 1, 1]);
        });

        test("slightly different values within tolerance", () => {
            const a = new TensorData(new Float64Array([1, 2, 3, 4]), [4]);
            const b = new TensorData(new Float64Array([1.005, 2.005, 3.005, 4.005]), [4]);
            const output = tensorFunctions.isClose(a, b);
            expect(Array.from(output.storage)).toEqual([1, 1, 1, 1]);
        });

        test("values outside tolerance", () => {
            const a = new TensorData(new Float64Array([1, 2, 3, 4]), [4]);
            const b = new TensorData(new Float64Array([1.02, 2.02, 3.02, 4.02]), [4]);
            const output = tensorFunctions.isClose(a, b);
            expect(Array.from(output.storage)).toEqual([0, 0, 0, 0]);
        });
    });
});

describe("Reduction Operations", () => {
    describe("sum", () => {
        test("sum along dim 0", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
            const output = tensorFunctions.sum(input, 0);
            expect(output.shape).toEqual([1, 3]);
            expect(Array.from(output.storage)).toEqual([5, 7, 9]);
        });

        test("sum along dim 1", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
            const output = tensorFunctions.sum(input, 1);
            expect(output.shape).toEqual([2, 1]);
            expect(Array.from(output.storage)).toEqual([6, 15]);
        });

        test("sum of 1D tensor", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4, 5]), [5]);
            const output = tensorFunctions.sum(input, 0);
            expect(output.shape).toEqual([1]);
            expect(output.storage[0]).toBe(15);
        });

        test("sum of 3D tensor along dim 1", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2, 2]);
            const output = tensorFunctions.sum(input, 1);
            expect(output.shape).toEqual([2, 1, 2]);
            expect(Array.from(output.storage)).toEqual([4, 6, 12, 14]);
        });
    });

    describe("prod", () => {
        test("prod along dim 0", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
            const output = tensorFunctions.prod(input, 0);
            expect(output.shape).toEqual([1, 2]);
            expect(Array.from(output.storage)).toEqual([3, 8]);
        });

        test("prod along dim 1", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
            const output = tensorFunctions.prod(input, 1);
            expect(output.shape).toEqual([2, 1]);
            expect(Array.from(output.storage)).toEqual([2, 12]);
        });

        test("prod with zeros", () => {
            const input = new TensorData(new Float64Array([1, 0, 3, 4]), [2, 2]);
            const output = tensorFunctions.prod(input, 1);
            expect(Array.from(output.storage)).toEqual([0, 12]);
        });
    });
});

describe("Shape Operations", () => {
    describe("permute", () => {
        test("permute 2D tensor", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
            const output = tensorFunctions.permute(input, [1, 0]);
            expect(output.shape).toEqual([3, 2]);
            // Original: [[1,2,3], [4,5,6]] -> Permuted: [[1,4], [2,5], [3,6]]
            expect(output.get([0, 0])).toBe(1);
            expect(output.get([0, 1])).toBe(4);
            expect(output.get([1, 0])).toBe(2);
            expect(output.get([1, 1])).toBe(5);
            expect(output.get([2, 0])).toBe(3);
            expect(output.get([2, 1])).toBe(6);
        });

        test("permute 3D tensor", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2, 2]);
            const output = tensorFunctions.permute(input, [2, 0, 1]);
            expect(output.shape).toEqual([2, 2, 2]);
        });

        test("permute is reversible", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
            const permuted = tensorFunctions.permute(input, [1, 0]);
            const restored = tensorFunctions.permute(permuted, [1, 0]);
            expect(restored.shape).toEqual([2, 3]);
            for (let i = 0; i < 2; i++) {
                for (let j = 0; j < 3; j++) {
                    expect(restored.get([i, j])).toBe(input.get([i, j]));
                }
            }
        });
    });

    describe("view", () => {
        test("view 1D to 2D", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [6]);
            const output = tensorFunctions.view(input, [2, 3]);
            expect(output.shape).toEqual([2, 3]);
            expect(Array.from(output.storage)).toEqual([1, 2, 3, 4, 5, 6]);
        });

        test("view 2D to 1D", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
            const output = tensorFunctions.view(input, [6]);
            expect(output.shape).toEqual([6]);
        });

        test("view 2D to different 2D", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
            const output = tensorFunctions.view(input, [3, 2]);
            expect(output.shape).toEqual([3, 2]);
        });

        test("view throws on size mismatch", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
            expect(() => tensorFunctions.view(input, [2, 2])).toThrow();
        });

        test("view throws on non-contiguous tensor", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
            const permuted = input.permute(1, 0); // Non-contiguous
            expect(() => tensorFunctions.view(permuted, [6])).toThrow();
        });
    });

    describe("contiguous", () => {
        test("contiguous tensor returns same", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
            const output = tensorFunctions.contiguous(input);
            expect(output).toBe(input);
        });

        test("non-contiguous tensor returns copy", () => {
            const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
            const permuted = input.permute(1, 0);
            const output = tensorFunctions.contiguous(permuted);
            
            expect(output).not.toBe(permuted);
            expect(output.shape).toEqual([3, 2]);
            // Should be contiguous now
            expect(output.strides).toEqual([2, 1]);
            
            // Values should match
            for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 2; j++) {
                    expect(output.get([i, j])).toBe(permuted.get([i, j]));
                }
            }
        });
    });
});
