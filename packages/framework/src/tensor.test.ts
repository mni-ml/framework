import { test, fc } from '@fast-check/jest';
import { describe, expect } from '@jest/globals';
import { Tensor } from './tensor.js';

// ============================================================
// Task 2.3 - Tensor Class Tests
// ============================================================

describe("Tensor Creation", () => {
    describe("Tensor.tensor", () => {
        test("create from scalar", () => {
            const t = Tensor.tensor(5);
            expect(t.shape).toEqual([]);
            expect(t.size).toBe(1);
            expect(t.dims).toBe(0);
            expect(t.item()).toBe(5);
        });

        test("create from 1D array", () => {
            const t = Tensor.tensor([1, 2, 3, 4]);
            expect(t.shape).toEqual([4]);
            expect(t.size).toBe(4);
            expect(t.dims).toBe(1);
        });

        test("create from 2D array", () => {
            const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            expect(t.shape).toEqual([2, 3]);
            expect(t.size).toBe(6);
            expect(t.dims).toBe(2);
        });

        test("create from 3D array", () => {
            const t = Tensor.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
            expect(t.shape).toEqual([2, 2, 2]);
            expect(t.size).toBe(8);
            expect(t.dims).toBe(3);
        });

        test("create with explicit shape", () => {
            const t = Tensor.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
            expect(t.shape).toEqual([2, 3]);
            expect(t.get([0, 0])).toBe(1);
            expect(t.get([1, 2])).toBe(6);
        });

        test("throws on shape mismatch", () => {
            expect(() => Tensor.tensor([1, 2, 3, 4], [2, 3])).toThrow();
        });
    });

    describe("Tensor.zeros", () => {
        test("creates zero tensor", () => {
            const t = Tensor.zeros([2, 3]);
            expect(t.shape).toEqual([2, 3]);
            expect(t.size).toBe(6);
            for (let i = 0; i < 2; i++) {
                for (let j = 0; j < 3; j++) {
                    expect(t.get([i, j])).toBe(0);
                }
            }
        });
    });

    describe("Tensor.ones", () => {
        test("creates ones tensor", () => {
            const t = Tensor.ones([2, 3]);
            expect(t.shape).toEqual([2, 3]);
            for (let i = 0; i < 2; i++) {
                for (let j = 0; j < 3; j++) {
                    expect(t.get([i, j])).toBe(1);
                }
            }
        });
    });

    describe("Tensor.rand", () => {
        test("creates random tensor in [0, 1)", () => {
            const t = Tensor.rand([10, 10]);
            expect(t.shape).toEqual([10, 10]);
            for (let i = 0; i < 10; i++) {
                for (let j = 0; j < 10; j++) {
                    const val = t.get([i, j]);
                    expect(val).toBeGreaterThanOrEqual(0);
                    expect(val).toBeLessThan(1);
                }
            }
        });
    });
});

describe("Tensor Properties", () => {
    test("size property", () => {
        expect(Tensor.tensor([1, 2, 3]).size).toBe(3);
        expect(Tensor.tensor([[1, 2], [3, 4]]).size).toBe(4);
        expect(Tensor.tensor(5).size).toBe(1);
    });

    test("dims property", () => {
        expect(Tensor.tensor([1, 2, 3]).dims).toBe(1);
        expect(Tensor.tensor([[1, 2], [3, 4]]).dims).toBe(2);
        expect(Tensor.tensor(5).dims).toBe(0);
    });

    test("shape property", () => {
        expect(Tensor.tensor([1, 2, 3]).shape).toEqual([3]);
        expect(Tensor.tensor([[1, 2, 3], [4, 5, 6]]).shape).toEqual([2, 3]);
    });
});

describe("Unary Operators", () => {
    describe("neg", () => {
        test("negates tensor", () => {
            const t = Tensor.tensor([1, -2, 3, -4]);
            const result = t.neg();
            expect(result.toArray()).toEqual([-1, 2, -3, 4]);
        });

        test("neg is self-inverse", () => {
            const t = Tensor.tensor([[1, 2], [3, 4]]);
            const result = t.neg().neg();
            expect(result.toArray()).toEqual([[1, 2], [3, 4]]);
        });
    });

    describe("sigmoid", () => {
        test("sigmoid of zeros", () => {
            const t = Tensor.tensor([0, 0, 0]);
            const result = t.sigmoid();
            result.toArray().forEach((v: number) => expect(v).toBeCloseTo(0.5));
        });

        test("sigmoid output in (0, 1)", () => {
            const t = Tensor.tensor([-10, -1, 0, 1, 10]);
            const result = t.sigmoid();
            result.toArray().forEach((v: number) => {
                expect(v).toBeGreaterThan(0);
                expect(v).toBeLessThan(1);
            });
        });
    });

    describe("relu", () => {
        test("relu of mixed values", () => {
            const t = Tensor.tensor([-2, -1, 0, 1, 2]);
            const result = t.relu();
            expect(result.toArray()).toEqual([0, 0, 0, 1, 2]);
        });
    });

    describe("log", () => {
        test("log of e is 1", () => {
            const t = Tensor.tensor([Math.E]);
            const result = t.log();
            expect(result.item()).toBeCloseTo(1);
        });
    });

    describe("exp", () => {
        test("exp of 0 is 1", () => {
            const t = Tensor.tensor([0]);
            const result = t.exp();
            expect(result.item()).toBeCloseTo(1);
        });

        test("exp(log(x)) = x", () => {
            const t = Tensor.tensor([1, 2, 3, 4]);
            const result = t.log().exp();
            const arr = result.toArray();
            expect(arr[0]).toBeCloseTo(1);
            expect(arr[1]).toBeCloseTo(2);
            expect(arr[2]).toBeCloseTo(3);
            expect(arr[3]).toBeCloseTo(4);
        });
    });

    describe("inv", () => {
        test("inv of 2 is 0.5", () => {
            const t = Tensor.tensor([2]);
            expect(t.inv().item()).toBeCloseTo(0.5);
        });
    });
});

describe("Binary Operators", () => {
    describe("add", () => {
        test("tensor + tensor", () => {
            const a = Tensor.tensor([[1, 2], [3, 4]]);
            const b = Tensor.tensor([[10, 20], [30, 40]]);
            const result = a.add(b);
            expect(result.toArray()).toEqual([[11, 22], [33, 44]]);
        });

        test("tensor + scalar", () => {
            const t = Tensor.tensor([1, 2, 3, 4]);
            const result = t.add(10);
            expect(result.toArray()).toEqual([11, 12, 13, 14]);
        });

        test("add with broadcasting", () => {
            const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            const b = Tensor.tensor([10, 20, 30]);
            const result = a.add(b);
            expect(result.toArray()).toEqual([[11, 22, 33], [14, 25, 36]]);
        });
    });

    describe("sub", () => {
        test("tensor - tensor", () => {
            const a = Tensor.tensor([10, 20, 30]);
            const b = Tensor.tensor([1, 2, 3]);
            const result = a.sub(b);
            expect(result.toArray()).toEqual([9, 18, 27]);
        });

        test("tensor - scalar", () => {
            const t = Tensor.tensor([10, 20, 30]);
            const result = t.sub(5);
            expect(result.toArray()).toEqual([5, 15, 25]);
        });
    });

    describe("mul", () => {
        test("tensor * tensor", () => {
            const a = Tensor.tensor([[1, 2], [3, 4]]);
            const b = Tensor.tensor([[2, 3], [4, 5]]);
            const result = a.mul(b);
            expect(result.toArray()).toEqual([[2, 6], [12, 20]]);
        });

        test("tensor * scalar", () => {
            const t = Tensor.tensor([1, 2, 3, 4]);
            const result = t.mul(2);
            expect(result.toArray()).toEqual([2, 4, 6, 8]);
        });

        test("mul with broadcasting", () => {
            const a = Tensor.tensor([[1, 2], [3, 4]]);
            const b = Tensor.tensor([10, 100]);
            const result = a.mul(b);
            expect(result.toArray()).toEqual([[10, 200], [30, 400]]);
        });
    });

    describe("lt", () => {
        test("less than comparison", () => {
            const a = Tensor.tensor([1, 5, 3, 7]);
            const b = Tensor.tensor([2, 4, 3, 8]);
            const result = a.lt(b);
            expect(result.toArray()).toEqual([1, 0, 0, 1]);
        });

        test("less than scalar", () => {
            const t = Tensor.tensor([1, 2, 3, 4, 5]);
            const result = t.lt(3);
            expect(result.toArray()).toEqual([1, 1, 0, 0, 0]);
        });
    });

    describe("eq", () => {
        test("equality comparison", () => {
            const a = Tensor.tensor([1, 2, 3, 4]);
            const b = Tensor.tensor([1, 5, 3, 8]);
            const result = a.eq(b);
            expect(result.toArray()).toEqual([1, 0, 1, 0]);
        });

        test("equality with scalar", () => {
            const t = Tensor.tensor([1, 2, 1, 2, 1]);
            const result = t.eq(1);
            expect(result.toArray()).toEqual([1, 0, 1, 0, 1]);
        });
    });

    describe("gt", () => {
        test("greater than comparison", () => {
            const a = Tensor.tensor([1, 5, 3, 7]);
            const b = Tensor.tensor([2, 4, 3, 8]);
            const result = a.gt(b);
            expect(result.toArray()).toEqual([0, 1, 0, 0]);
        });

        test("greater than scalar", () => {
            const t = Tensor.tensor([1, 2, 3, 4, 5]);
            const result = t.gt(3);
            expect(result.toArray()).toEqual([0, 0, 0, 1, 1]);
        });
    });

    describe("is_close", () => {
        test("identical tensors are close", () => {
            const a = Tensor.tensor([1, 2, 3, 4]);
            const b = Tensor.tensor([1, 2, 3, 4]);
            const result = a.is_close(b);
            expect(result.toArray()).toEqual([1, 1, 1, 1]);
        });

        test("slightly different within tolerance", () => {
            const a = Tensor.tensor([1, 2, 3, 4]);
            const b = Tensor.tensor([1.005, 2.005, 3.005, 4.005]);
            const result = a.is_close(b);
            expect(result.toArray()).toEqual([1, 1, 1, 1]);
        });
    });

    describe("radd and rmul", () => {
        test("radd is same as add", () => {
            const t = Tensor.tensor([1, 2, 3]);
            expect(t.radd(10).toArray()).toEqual([11, 12, 13]);
        });

        test("rmul is same as mul", () => {
            const t = Tensor.tensor([1, 2, 3]);
            expect(t.rmul(2).toArray()).toEqual([2, 4, 6]);
        });
    });
});

describe("Reduction Operators", () => {
    describe("sum", () => {
        test("sum along dim 0", () => {
            const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            const result = t.sum(0);
            expect(result.shape).toEqual([1, 3]);
            expect(result.toArray()).toEqual([[5, 7, 9]]);
        });

        test("sum along dim 1", () => {
            const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            const result = t.sum(1);
            expect(result.shape).toEqual([2, 1]);
            expect(result.toArray()).toEqual([[6], [15]]);
        });

        test("sum all elements (no dim)", () => {
            const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            const result = t.sum();
            expect(result.item()).toBe(21);
        });

        test("sum 1D tensor", () => {
            const t = Tensor.tensor([1, 2, 3, 4, 5]);
            const result = t.sum();
            expect(result.item()).toBe(15);
        });

        test("sum 3D tensor along dim 1", () => {
            const t = Tensor.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
            const result = t.sum(1);
            expect(result.shape).toEqual([2, 1, 2]);
            expect(result.toArray()).toEqual([[[4, 6]], [[12, 14]]]);
        });
    });

    describe("mean", () => {
        test("mean along dim 0", () => {
            const t = Tensor.tensor([[2, 4], [6, 8]]);
            const result = t.mean(0);
            expect(result.shape).toEqual([1, 2]);
            expect(result.toArray()).toEqual([[4, 6]]);
        });

        test("mean along dim 1", () => {
            const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            const result = t.mean(1);
            expect(result.shape).toEqual([2, 1]);
            expect(result.toArray()).toEqual([[2], [5]]);
        });

        test("mean all elements (no dim)", () => {
            const t = Tensor.tensor([[1, 2], [3, 4]]);
            const result = t.mean();
            expect(result.item()).toBeCloseTo(2.5);
        });
    });

    describe("all", () => {
        test("all true (non-zero)", () => {
            const t = Tensor.tensor([1, 2, 3, 4]);
            const result = t.all();
            expect(result.item()).toBe(1);
        });

        test("all false (has zero)", () => {
            const t = Tensor.tensor([1, 0, 3, 4]);
            const result = t.all();
            expect(result.item()).toBe(0);
        });

        test("all along dim", () => {
            const t = Tensor.tensor([[1, 0], [1, 1]]);
            const result = t.all(1);
            expect(result.shape).toEqual([2, 1]);
            expect(result.toArray()).toEqual([[0], [1]]);
        });

        test("all with 2D tensor", () => {
            const t = Tensor.tensor([[1, 2], [3, 4]]);
            const result = t.all();
            expect(result.item()).toBe(1);
        });
    });
});

describe("Shape Operations", () => {
    describe("permute", () => {
        test("permute 2D tensor", () => {
            const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            const result = t.permute(1, 0);
            expect(result.shape).toEqual([3, 2]);
            expect(result.toArray()).toEqual([[1, 4], [2, 5], [3, 6]]);
        });

        test("permute is reversible", () => {
            const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            const result = t.permute(1, 0).permute(1, 0);
            expect(result.shape).toEqual([2, 3]);
            expect(result.toArray()).toEqual([[1, 2, 3], [4, 5, 6]]);
        });
    });

    describe("view", () => {
        test("view 1D to 2D", () => {
            const t = Tensor.tensor([1, 2, 3, 4, 5, 6]);
            const result = t.view(2, 3);
            expect(result.shape).toEqual([2, 3]);
            expect(result.toArray()).toEqual([[1, 2, 3], [4, 5, 6]]);
        });

        test("view 2D to 1D", () => {
            const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            const result = t.view(6);
            expect(result.shape).toEqual([6]);
            expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
        });

        test("view throws on size mismatch", () => {
            const t = Tensor.tensor([1, 2, 3, 4, 5, 6]);
            expect(() => t.view(2, 2)).toThrow();
        });
    });

    describe("contiguous", () => {
        test("contiguous tensor stays same", () => {
            const t = Tensor.tensor([[1, 2], [3, 4]]);
            const result = t.contiguous();
            expect(result.data).toBe(t.data);
        });

        test("non-contiguous becomes contiguous", () => {
            const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            const permuted = t.permute(1, 0);
            const result = permuted.contiguous();
            expect(result.data).not.toBe(permuted.data);
            expect(result.toArray()).toEqual([[1, 4], [2, 5], [3, 6]]);
        });
    });
});

describe("Gradient Utilities", () => {
    describe("zero_grad_", () => {
        test("sets grad to null", () => {
            const t = Tensor.tensor([1, 2, 3]);
            t.grad = Tensor.tensor([1, 1, 1]);
            expect(t.grad).not.toBeNull();
            t.zero_grad_();
            expect(t.grad).toBeNull();
        });
    });
});

describe("Indexing and Inspection", () => {
    describe("get and set", () => {
        test("get value at index", () => {
            const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            expect(t.get([0, 0])).toBe(1);
            expect(t.get([0, 2])).toBe(3);
            expect(t.get([1, 1])).toBe(5);
        });

        test("set value at index", () => {
            const t = Tensor.tensor([[1, 2], [3, 4]]);
            t.set([0, 1], 100);
            expect(t.get([0, 1])).toBe(100);
        });
    });

    describe("item", () => {
        test("item returns scalar value", () => {
            const t = Tensor.tensor(42);
            expect(t.item()).toBe(42);
        });

        test("item works for single element tensor", () => {
            const t = Tensor.tensor([[5]]);
            expect(t.item()).toBe(5);
        });

        test("item throws for multi-element tensor", () => {
            const t = Tensor.tensor([1, 2, 3]);
            expect(() => t.item()).toThrow();
        });
    });

    describe("toArray", () => {
        test("toArray for 1D", () => {
            const t = Tensor.tensor([1, 2, 3]);
            expect(t.toArray()).toEqual([1, 2, 3]);
        });

        test("toArray for 2D", () => {
            const t = Tensor.tensor([[1, 2], [3, 4]]);
            expect(t.toArray()).toEqual([[1, 2], [3, 4]]);
        });

        test("toArray for 3D", () => {
            const t = Tensor.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
            expect(t.toArray()).toEqual([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        });
    });

    describe("toString", () => {
        test("toString for scalar", () => {
            const t = Tensor.tensor(5);
            expect(t.toString()).toBe("Tensor(5)");
        });

        test("toString for tensor", () => {
            const t = Tensor.tensor([1, 2, 3]);
            expect(t.toString()).toContain("Tensor(");
            expect(t.toString()).toContain("[1,2,3]");
            expect(t.toString()).toContain("shape=[3]");
        });
    });
});

describe("Property-based tests", () => {
    const smallDim = fc.integer({ min: 1, max: 4 });
    const shape2D = fc.tuple(smallDim, smallDim);

    test.prop([shape2D])("add is commutative", (shape) => {
        const a = Tensor.rand(shape);
        const b = Tensor.rand(shape);
        const ab = a.add(b);
        const ba = b.add(a);
        
        for (let i = 0; i < shape[0]; i++) {
            for (let j = 0; j < shape[1]; j++) {
                expect(ab.get([i, j])).toBeCloseTo(ba.get([i, j]));
            }
        }
    });

    test.prop([shape2D])("mul is commutative", (shape) => {
        const a = Tensor.rand(shape);
        const b = Tensor.rand(shape);
        const ab = a.mul(b);
        const ba = b.mul(a);
        
        for (let i = 0; i < shape[0]; i++) {
            for (let j = 0; j < shape[1]; j++) {
                expect(ab.get([i, j])).toBeCloseTo(ba.get([i, j]));
            }
        }
    });

    test.prop([shape2D])("adding zero is identity", (shape) => {
        const t = Tensor.rand(shape);
        const result = t.add(0);
        
        for (let i = 0; i < shape[0]; i++) {
            for (let j = 0; j < shape[1]; j++) {
                expect(result.get([i, j])).toBeCloseTo(t.get([i, j]));
            }
        }
    });

    test.prop([shape2D])("multiplying by one is identity", (shape) => {
        const t = Tensor.rand(shape);
        const result = t.mul(1);
        
        for (let i = 0; i < shape[0]; i++) {
            for (let j = 0; j < shape[1]; j++) {
                expect(result.get([i, j])).toBeCloseTo(t.get([i, j]));
            }
        }
    });

    test.prop([shape2D])("neg is self-inverse", (shape) => {
        const t = Tensor.rand(shape);
        const result = t.neg().neg();
        
        for (let i = 0; i < shape[0]; i++) {
            for (let j = 0; j < shape[1]; j++) {
                expect(result.get([i, j])).toBeCloseTo(t.get([i, j]));
            }
        }
    });

    test.prop([shape2D])("sum of all equals sum of row sums", (shape) => {
        const t = Tensor.rand(shape);
        const totalSum = t.sum().item();
        
        // Sum along dim 1 first, then sum result
        const rowSums = t.sum(1);
        const sumOfRowSums = rowSums.sum().item();
        
        expect(totalSum).toBeCloseTo(sumOfRowSums);
    });
});

// ============================================================
// Edge Case Tests
// ============================================================

describe("Edge Cases", () => {
    describe("Scalar (0-d) tensor operations", () => {
        test("sum on scalar returns scalar", () => {
            const t = Tensor.tensor(5);
            const result = t.sum();
            expect(result.dims).toBe(0);
            expect(result.item()).toBe(5);
        });

        test("mean on scalar returns scalar", () => {
            const t = Tensor.tensor(10);
            const result = t.mean();
            expect(result.item()).toBeCloseTo(10);
        });

        test("all on scalar (truthy)", () => {
            const t = Tensor.tensor(5);
            expect(t.all().item()).toBe(1);
        });

        test("all on scalar (falsy)", () => {
            const t = Tensor.tensor(0);
            expect(t.all().item()).toBe(0);
        });

        test("neg on scalar", () => {
            const t = Tensor.tensor(5);
            expect(t.neg().item()).toBe(-5);
        });

        test("add scalar tensors", () => {
            const a = Tensor.tensor(3);
            const b = Tensor.tensor(4);
            expect(a.add(b).item()).toBe(7);
        });

        test("mul scalar tensors", () => {
            const a = Tensor.tensor(3);
            const b = Tensor.tensor(4);
            expect(a.mul(b).item()).toBe(12);
        });

        test("toArray on scalar", () => {
            const t = Tensor.tensor(42);
            expect(t.toArray()).toBe(42);
        });
    });

    describe("Single element tensor operations", () => {
        test("item on [1] shape tensor", () => {
            const t = Tensor.tensor([5]);
            expect(t.item()).toBe(5);
        });

        test("item on [1,1] shape tensor", () => {
            const t = Tensor.tensor([[5]]);
            expect(t.item()).toBe(5);
        });

        test("item on [1,1,1] shape tensor", () => {
            const t = Tensor.tensor([[[5]]]);
            expect(t.item()).toBe(5);
        });

        test("sum on [1] tensor", () => {
            const t = Tensor.tensor([5]);
            expect(t.sum().item()).toBe(5);
        });

        test("mean on [1] tensor", () => {
            const t = Tensor.tensor([5]);
            expect(t.mean().item()).toBe(5);
        });
    });

    describe("Dimension validation", () => {
        test("sum throws on invalid negative dim", () => {
            const t = Tensor.tensor([[1, 2], [3, 4]]);
            expect(() => t.sum(-1)).toThrow();
        });

        test("sum throws on dim >= dims", () => {
            const t = Tensor.tensor([[1, 2], [3, 4]]);
            expect(() => t.sum(2)).toThrow();
        });

        test("mean throws on invalid dim", () => {
            const t = Tensor.tensor([[1, 2], [3, 4]]);
            expect(() => t.mean(-1)).toThrow();
            expect(() => t.mean(2)).toThrow();
        });

        test("all throws on invalid dim", () => {
            const t = Tensor.tensor([[1, 2], [3, 4]]);
            expect(() => t.all(-1)).toThrow();
            expect(() => t.all(2)).toThrow();
        });
    });

    describe("Non-contiguous tensor operations", () => {
        test("operations on permuted tensor", () => {
            const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            const permuted = t.permute(1, 0); // shape [3, 2]
            
            // neg should work
            const negated = permuted.neg();
            expect(negated.shape).toEqual([3, 2]);
            expect(negated.get([0, 0])).toBe(-1);
            expect(negated.get([0, 1])).toBe(-4);
        });

        test("sum on permuted tensor", () => {
            const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            const permuted = t.permute(1, 0); // shape [3, 2]: [[1,4], [2,5], [3,6]]
            
            // Sum along dim 0
            const summed = permuted.sum(0);
            expect(summed.shape).toEqual([1, 2]);
            // [1+2+3, 4+5+6] = [6, 15]
            expect(summed.toArray()).toEqual([[6, 15]]);
        });

        test("view on permuted tensor throws", () => {
            const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            const permuted = t.permute(1, 0);
            expect(() => permuted.view(6)).toThrow();
        });

        test("contiguous then view works", () => {
            const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            const permuted = t.permute(1, 0);
            const contig = permuted.contiguous();
            const viewed = contig.view(6);
            expect(viewed.shape).toEqual([6]);
            expect(viewed.toArray()).toEqual([1, 4, 2, 5, 3, 6]);
        });

        test("item on permuted single-element tensor", () => {
            const t = Tensor.tensor([[5]]);
            const permuted = t.permute(1, 0);
            expect(permuted.item()).toBe(5);
        });
    });

    describe("Broadcasting edge cases", () => {
        test("scalar broadcasts to any shape", () => {
            const scalar = Tensor.tensor(5);
            const t = Tensor.tensor([[1, 2], [3, 4]]);
            const result = t.add(scalar);
            expect(result.toArray()).toEqual([[6, 7], [8, 9]]);
        });

        test("3D + 1D broadcasting", () => {
            const a = Tensor.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]); // [2, 2, 2]
            const b = Tensor.tensor([10, 100]); // [2]
            const result = a.add(b);
            expect(result.shape).toEqual([2, 2, 2]);
            expect(result.get([0, 0, 0])).toBe(11);
            expect(result.get([0, 0, 1])).toBe(102);
        });

        test("2D + scalar number (not tensor)", () => {
            const t = Tensor.tensor([[1, 2], [3, 4]]);
            const result = t.add(10);
            expect(result.toArray()).toEqual([[11, 12], [13, 14]]);
        });
    });

    describe("Large tensor operations", () => {
        test("sum of large tensor", () => {
            const size = 1000;
            const t = Tensor.ones([size]);
            expect(t.sum().item()).toBe(size);
        });

        test("mean of large tensor", () => {
            const t = Tensor.ones([100, 100]);
            expect(t.mean().item()).toBeCloseTo(1);
        });
    });

    describe("Chained operations", () => {
        test("chain of unary ops", () => {
            const t = Tensor.tensor([1, 2, 3, 4]);
            const result = t.neg().neg().neg();
            expect(result.toArray()).toEqual([-1, -2, -3, -4]);
        });

        test("chain of binary ops", () => {
            const t = Tensor.tensor([1, 2, 3, 4]);
            const result = t.add(1).mul(2).sub(1);
            expect(result.toArray()).toEqual([3, 5, 7, 9]);
        });

        test("chain reductions", () => {
            const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            const result = t.sum(1).sum(0);
            expect(result.item()).toBe(21);
        });
    });

    describe("Zero and negative values", () => {
        test("relu on negative values", () => {
            const t = Tensor.tensor([-5, -1, 0, 1, 5]);
            expect(t.relu().toArray()).toEqual([0, 0, 0, 1, 5]);
        });

        test("log of 1", () => {
            const t = Tensor.tensor([1]);
            expect(t.log().item()).toBeCloseTo(0);
        });

        test("exp of 0", () => {
            const t = Tensor.tensor([0]);
            expect(t.exp().item()).toBeCloseTo(1);
        });

        test("sigmoid of 0", () => {
            const t = Tensor.tensor([0]);
            expect(t.sigmoid().item()).toBeCloseTo(0.5);
        });
    });

    describe("Reduction with size-1 dimensions", () => {
        test("sum along already-size-1 dimension", () => {
            const t = Tensor.tensor([[1, 2, 3]]); // shape [1, 3]
            const result = t.sum(0);
            expect(result.shape).toEqual([1, 3]);
            expect(result.toArray()).toEqual([[1, 2, 3]]);
        });

        test("mean along size-1 dimension", () => {
            const t = Tensor.tensor([[1], [2], [3]]); // shape [3, 1]
            const result = t.mean(1);
            expect(result.shape).toEqual([3, 1]);
            expect(result.toArray()).toEqual([[1], [2], [3]]);
        });
    });

    describe("toArray correctness", () => {
        test("toArray called multiple times gives same result", () => {
            const t = Tensor.tensor([[1, 2], [3, 4]]);
            const arr1 = t.toArray();
            const arr2 = t.toArray();
            expect(arr1).toEqual(arr2);
            expect(arr1).toEqual([[1, 2], [3, 4]]);
        });

        test("toArray on permuted tensor", () => {
            const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
            const permuted = t.permute(1, 0);
            expect(permuted.toArray()).toEqual([[1, 4], [2, 5], [3, 6]]);
        });
    });
});
