import { test, fc } from '@fast-check/jest';
import { describe, expect } from '@jest/globals';
import { centralDifference, Context, topologicalSort, backPropagate } from "./autodiff.js";
import { Scalar } from "./scalar.js"
import { ScalarFunction, ScalarHistory } from "./scalar_functions.js"; 
import * as operators from './operators.js';
const DIGIT_TOLERANCE = 4;

/** floats from set of small finite floats */
const smallFloat = fc.double({ noNaN: true, min: -100, max: 100 });

/** Positive floats for functions like log, sqrt */
const positiveFloat = fc.double({ noNaN: true, min: 0.1, max: 100 });

/** Non-zero floats for division */
const nonZeroFloat = fc.double({ noNaN: true, min: -100, max: 100 }).filter(x => Math.abs(x) > 0.1);

function assertClose(actual: number, expected: number, digits = DIGIT_TOLERANCE) {
  expect(actual).toBeCloseTo(expected, digits);
}

// ============================================================
// Central Difference Tests
// ============================================================

describe("centralDifference", () => {
  describe("single variable functions", () => {
    test("derivative of f(x) = x is 1", () => {
      fc.assert(
        fc.property(smallFloat, (x) => {
          const f = (a: number) => a;
          assertClose(centralDifference(f, [x]), 1);
        })
      );
    });

    test("derivative of f(x) = x^2 is 2x", () => {
      fc.assert(
        fc.property(smallFloat, (x) => {
          const f = (a: number) => a * a;
          assertClose(centralDifference(f, [x]), 2 * x);
        })
      );
    });

    test("derivative of f(x) = x^3 is 3x^2", () => {
      fc.assert(
        fc.property(smallFloat, (x) => {
          const f = (a: number) => a * a * a;
          assertClose(centralDifference(f, [x]), 3 * x * x, 3);
        })
      );
    });

    test("derivative of f(x) = 5x + 3 is 5", () => {
      fc.assert(
        fc.property(smallFloat, (x) => {
          const f = (a: number) => 5 * a + 3;
          assertClose(centralDifference(f, [x]), 5);
        })
      );
    });

    test("derivative of constant f(x) = 7 is 0", () => {
      fc.assert(
        fc.property(smallFloat, (x) => {
          const f = (_a: number) => 7;
          assertClose(centralDifference(f, [x]), 0);
        })
      );
    });

    test("derivative of f(x) = sin(x) is cos(x)", () => {
      fc.assert(
        fc.property(smallFloat, (x) => {
          const f = (a: number) => Math.sin(a);
          assertClose(centralDifference(f, [x]), Math.cos(x));
        })
      );
    });

    test("derivative of f(x) = cos(x) is -sin(x)", () => {
      fc.assert(
        fc.property(smallFloat, (x) => {
          const f = (a: number) => Math.cos(a);
          assertClose(centralDifference(f, [x]), -Math.sin(x));
        })
      );
    });

    test("derivative of f(x) = e^x is e^x", () => {
      // Use smaller range to avoid overflow
      const boundedFloat = fc.double({ noNaN: true, min: -10, max: 10 });
      fc.assert(
        fc.property(boundedFloat, (x) => {
          const f = (a: number) => Math.exp(a);
          assertClose(centralDifference(f, [x]), Math.exp(x));
        })
      );
    });

    test("derivative of f(x) = log(x) is 1/x", () => {
      fc.assert(
        fc.property(positiveFloat, (x) => {
          const f = (a: number) => Math.log(a);
          assertClose(centralDifference(f, [x]), 1 / x);
        })
      );
    });

    test("derivative of f(x) = 1/x is -1/x^2", () => {
      fc.assert(
        fc.property(nonZeroFloat, (x) => {
          const f = (a: number) => 1 / a;
          assertClose(centralDifference(f, [x]), -1 / (x * x), 3);
        })
      );
    });

    test("derivative of f(x) = sqrt(x) is 1/(2*sqrt(x))", () => {
      fc.assert(
        fc.property(positiveFloat, (x) => {
          const f = (a: number) => Math.sqrt(a);
          assertClose(centralDifference(f, [x]), 1 / (2 * Math.sqrt(x)));
        })
      );
    });
  });

  describe("multi-variable functions - partial derivatives", () => {
    test("partial derivative of f(x,y) = x + y with respect to x is 1", () => {
      fc.assert(
        fc.property(smallFloat, smallFloat, (x, y) => {
          const f = (a: number, b: number) => a + b;
          assertClose(centralDifference(f, [x, y], 0), 1);  // df/dx
        })
      );
    });

    test("partial derivative of f(x,y) = x + y with respect to y is 1", () => {
      fc.assert(
        fc.property(smallFloat, smallFloat, (x, y) => {
          const f = (a: number, b: number) => a + b;
          assertClose(centralDifference(f, [x, y], 1), 1);  // df/dy
        })
      );
    });

    test("partial derivative of f(x,y) = x * y with respect to x is y", () => {
      fc.assert(
        fc.property(smallFloat, smallFloat, (x, y) => {
          const f = (a: number, b: number) => a * b;
          assertClose(centralDifference(f, [x, y], 0), y);  // df/dx = y
        })
      );
    });

    test("partial derivative of f(x,y) = x * y with respect to y is x", () => {
      fc.assert(
        fc.property(smallFloat, smallFloat, (x, y) => {
          const f = (a: number, b: number) => a * b;
          assertClose(centralDifference(f, [x, y], 1), x);  // df/dy = x
        })
      );
    });

    test("partial derivative of f(x,y) = x^2 + y^2 with respect to x is 2x", () => {
      fc.assert(
        fc.property(smallFloat, smallFloat, (x, y) => {
          const f = (a: number, b: number) => a * a + b * b;
          assertClose(centralDifference(f, [x, y], 0), 2 * x);  // df/dx = 2x
        })
      );
    });

    test("partial derivative of f(x,y) = x^2 + y^2 with respect to y is 2y", () => {
      fc.assert(
        fc.property(smallFloat, smallFloat, (x, y) => {
          const f = (a: number, b: number) => a * a + b * b;
          assertClose(centralDifference(f, [x, y], 1), 2 * y);  // df/dy = 2y
        })
      );
    });

    test("partial derivative of f(x,y,z) = x*y*z with respect to each variable", () => {
      fc.assert(
        fc.property(smallFloat, smallFloat, smallFloat, (x, y, z) => {
          const f = (a: number, b: number, c: number) => a * b * c;
          assertClose(centralDifference(f, [x, y, z], 0), y * z, 3);  // df/dx = y*z
          assertClose(centralDifference(f, [x, y, z], 1), x * z, 3);  // df/dy = x*z
          assertClose(centralDifference(f, [x, y, z], 2), x * y, 3);  // df/dz = x*y
        })
      );
    });
  });

  describe("epsilon parameter", () => {
    test("smaller epsilon gives more accurate results for smooth functions", () => {
      const x = 2;
      const f = (a: number) => a * a;  // derivative is 2x = 4
      const expected = 4;

      const result1 = centralDifference(f, [x], 0, 1e-3);
      const result2 = centralDifference(f, [x], 0, 1e-6);
      const result3 = centralDifference(f, [x], 0, 1e-9);

      // All should be close, but smaller epsilon generally more accurate
      assertClose(result1, expected, 3);
      assertClose(result2, expected, 6);
      assertClose(result3, expected, 6);  // Very small epsilon may have floating point issues
    });
  });
});

// ============================================================
// Context Tests
// ============================================================

describe("Context", () => {
  test("saveForBackward stores values", () => {
    const ctx = new Context();
    ctx.saveForBackward(1, 2, 3);
    expect(ctx.savedValues).toEqual([1, 2, 3]);
  });

  test("savedValues returns empty array initially", () => {
    const ctx = new Context();
    expect(ctx.savedValues).toEqual([]);
  });

  test("saveForBackward overwrites previous values", () => {
    const ctx = new Context();
    ctx.saveForBackward(1, 2);
    ctx.saveForBackward(5, 6, 7);
    expect(ctx.savedValues).toEqual([5, 6, 7]);
  });

  test("saveForBackward handles single value", () => {
    const ctx = new Context();
    ctx.saveForBackward(42);
    expect(ctx.savedValues).toEqual([42]);
  });

  test("saveForBackward handles no values", () => {
    const ctx = new Context();
    ctx.saveForBackward();
    expect(ctx.savedValues).toEqual([]);
  });
});

// ============================================================
// Autodiff Tests
// ============================================================


export class Function1 extends ScalarFunction {
    static forward(ctx: Context, x: number, y: number): number {
        return operators.add(x, y);
    }

    static backward(ctx: Context, dOut: number): [number, number] {
      return [dOut, dOut];
    }
}

export class Function2 extends ScalarFunction {
    static forward(ctx: Context, x: number, y: number): number {
        ctx.saveForBackward(x, y);
        return x * y + x;
    }

    static backward(ctx: Context, dOut: number): [number, number] {
      const [x, y] = ctx.savedValues;
      return [dOut * (y + 1), dOut * x];
    }
}

// ============================================================
// Chain Rule Tests
// ============================================================

describe("Chain rule", () => {
  test("length of scalar:gradient array is 2", () => {
    const x = new Scalar(0.0);
    const constant = new Scalar(0.0, new ScalarHistory(Function1, new Context(), [x,x]));
    const back = constant.chainRule(5);

    expect(Array.from(back).length).toEqual(2);
  })

  test("derivative correctness", () => {
    const v = new Scalar(0.0, new ScalarHistory());
    const constant = new Scalar(0.0, new ScalarHistory(Function1, new Context(), [v,v]));
    const back = constant.chainRule(5);
    const backArr = Array.from(back);

    expect(backArr.length).toEqual(2);

    const [_, deriv] = backArr[0]!;
    expect(deriv).toEqual(5);
  })

  test("constants ignored, variables get derivatives", () => {
    const v = new Scalar(5);
    const constant = 10;
    const y = Scalar.apply(Function2, constant, v);
    
    const back = y.chainRule(5);
    const backArr = Array.from(back);

    expect(backArr.length).toEqual(2);

    const [variable, deriv] = backArr[1]!;
    expect(variable.name).toEqual(v.name);
    expect(deriv).toEqual(5 * 10);
  })

  test("rule3: constants ignored, variables get derivatives", () => {
    const constant = 10;
    const v = new Scalar(5);

    const y = Scalar.apply(Function2, constant, v);

    const back = y.chainRule(5);
    const backArr = Array.from(back);

    expect(backArr.length).toEqual(2);

    // second input is the variable `v`
    const [variable, deriv] = backArr[1]!;
    expect(variable.name).toEqual(v.name);
    expect(deriv).toEqual(5 * 10);
  });

  test("rule4: two variables get correct derivatives", () => {
    const v1 = new Scalar(5);
    const v2 = new Scalar(10);

    const y = Scalar.apply(Function2, v1, v2);

    const back = y.chainRule(5);
    const backArr = Array.from(back);

    expect(backArr.length).toEqual(2);

    // grad wrt v1 (x) is dOut * (y + 1)
    let [variable, deriv] = backArr[0]!;
    expect(variable.name).toEqual(v1.name);
    expect(deriv).toEqual(5 * (10 + 1));

    // grad wrt v2 (y) is dOut * x
    [variable, deriv] = backArr[1]!;
    expect(variable.name).toEqual(v2.name);
    expect(deriv).toEqual(5 * 5);
  });

})

// ============================================================
// Topological Sort Tests (task1_4)
// ============================================================

describe("topologicalSort", () => {
  test("single leaf node returns just that node", () => {
    const x = new Scalar(5);
    const sorted = topologicalSort(x);
    
    expect(sorted.length).toEqual(1);
    expect(sorted[0]).toBe(x);
  });

  test("simple operation returns output before inputs", () => {
    const x = new Scalar(3);
    const y = new Scalar(4);
    const z = x.mul(y);
    
    const sorted = topologicalSort(z);
    
    // z should come before x and y
    expect(sorted.length).toEqual(3);
    expect(sorted[0]).toBe(z);
    // x and y should both be in the list after z
    expect(sorted.slice(1)).toContain(x);
    expect(sorted.slice(1)).toContain(y);
  });

  test("chain of operations has correct order", () => {
    const x = new Scalar(2);
    const y = x.mul(3);      // y = x * 3
    const z = y.add(1);      // z = y + 1
    
    const sorted = topologicalSort(z);
    
    // z should come first, then y, then leaves
    const zIdx = sorted.indexOf(z);
    const yIdx = sorted.indexOf(y);
    const xIdx = sorted.indexOf(x);
    
    expect(zIdx).toBeLessThan(yIdx);
    expect(yIdx).toBeLessThan(xIdx);
  });

  test("variable used multiple times appears only once", () => {
    const x = new Scalar(3);
    const z = x.mul(x);  // x * x
    
    const sorted = topologicalSort(z);
    
    // x should appear exactly once
    const xCount = sorted.filter(s => s === x).length;
    expect(xCount).toEqual(1);
    expect(sorted.length).toEqual(2);  // z and x
  });

  test("diamond dependency graph", () => {
    const x = new Scalar(2);
    const a = x.mul(3);      // a = x * 3
    const b = x.add(1);      // b = x + 1
    const c = a.mul(b);      // c = a * b
    
    const sorted = topologicalSort(c);
    
    // c should come first
    expect(sorted[0]).toBe(c);
    
    // a and b should come before x
    const aIdx = sorted.indexOf(a);
    const bIdx = sorted.indexOf(b);
    const xIdx = sorted.indexOf(x);
    
    expect(aIdx).toBeLessThan(xIdx);
    expect(bIdx).toBeLessThan(xIdx);
  });
});

// ============================================================
// Backpropagate Tests (task1_4)
// ============================================================

describe("backPropagate", () => {
  test("simple multiplication: z = x * y", () => {
    const x = new Scalar(3);
    const y = new Scalar(4);
    const z = x.mul(y);
    
    backPropagate(z, 1.0);
    
    // dz/dx = y = 4
    expect(x.derivative).toBeCloseTo(4, 5);
    // dz/dy = x = 3
    expect(y.derivative).toBeCloseTo(3, 5);
  });

  test("simple addition: z = x + y", () => {
    const x = new Scalar(3);
    const y = new Scalar(4);
    const z = x.add(y);
    
    backPropagate(z, 1.0);
    
    // dz/dx = 1
    expect(x.derivative).toBeCloseTo(1, 5);
    // dz/dy = 1
    expect(y.derivative).toBeCloseTo(1, 5);
  });

  test("variable used twice: z = x * x (should accumulate)", () => {
    const x = new Scalar(3);
    const z = x.mul(x);
    
    backPropagate(z, 1.0);
    
    // d(x^2)/dx = 2x = 6
    expect(x.derivative).toBeCloseTo(6, 5);
  });

  test("chain: z = (x * y) + x", () => {
    const x = new Scalar(3);
    const y = new Scalar(4);
    const z = x.mul(y).add(x);
    
    backPropagate(z, 1.0);
    
    // z = x*y + x, dz/dx = y + 1 = 5
    expect(x.derivative).toBeCloseTo(5, 5);
    // dz/dy = x = 3
    expect(y.derivative).toBeCloseTo(3, 5);
  });

  test("log function: z = log(x)", () => {
    const x = new Scalar(2);
    const z = x.log();
    
    backPropagate(z, 1.0);
    
    // d(log(x))/dx = 1/x = 0.5
    expect(x.derivative).toBeCloseTo(0.5, 5);
  });

  test("exp function: z = exp(x)", () => {
    const x = new Scalar(1);
    const z = x.exp();
    
    backPropagate(z, 1.0);
    
    // d(e^x)/dx = e^x = e
    expect(x.derivative).toBeCloseTo(Math.E, 5);
  });

  test("sigmoid function: z = sigmoid(x)", () => {
    const x = new Scalar(0);
    const z = x.sigmoid();
    
    backPropagate(z, 1.0);
    
    // sigmoid(0) = 0.5, d(sigmoid)/dx = sigmoid * (1 - sigmoid) = 0.25
    expect(x.derivative).toBeCloseTo(0.25, 5);
  });

  test("relu function: z = relu(x) for positive x", () => {
    const x = new Scalar(5);
    const z = x.relu();
    
    backPropagate(z, 1.0);
    
    // d(relu(x))/dx = 1 for x > 0
    expect(x.derivative).toBeCloseTo(1, 5);
  });

  test("relu function: z = relu(x) for negative x", () => {
    const x = new Scalar(-5);
    const z = x.relu();
    
    backPropagate(z, 1.0);
    
    // d(relu(x))/dx = 0 for x < 0
    expect(x.derivative).toBeCloseTo(0, 5);
  });

  test("negation: z = -x", () => {
    const x = new Scalar(3);
    const z = x.neg();
    
    backPropagate(z, 1.0);
    
    // d(-x)/dx = -1
    expect(x.derivative).toBeCloseTo(-1, 5);
  });

  test("division: z = x / y", () => {
    const x = new Scalar(6);
    const y = new Scalar(2);
    const z = x.div(y);
    
    backPropagate(z, 1.0);
    
    // d(x/y)/dx = 1/y = 0.5
    expect(x.derivative).toBeCloseTo(0.5, 5);
    // d(x/y)/dy = -x/y^2 = -6/4 = -1.5
    expect(y.derivative).toBeCloseTo(-1.5, 5);
  });

  test("complex expression matches numerical gradient", () => {
    // f(x, y) = x * y + x^2
    const f = (xVal: number, yVal: number) => {
      const x = new Scalar(xVal);
      const y = new Scalar(yVal);
      return x.mul(y).add(x.mul(x));
    };
    
    const xVal = 3;
    const yVal = 4;
    
    // Compute analytical gradients
    const x = new Scalar(xVal);
    const y = new Scalar(yVal);
    const z = x.mul(y).add(x.mul(x));
    backPropagate(z, 1.0);
    
    // Compute numerical gradients
    const numericFn = (a: number, b: number) => a * b + a * a;
    const numDx = centralDifference(numericFn, [xVal, yVal], 0);
    const numDy = centralDifference(numericFn, [xVal, yVal], 1);
    
    expect(x.derivative).toBeCloseTo(numDx, 4);
    expect(y.derivative).toBeCloseTo(numDy, 4);
  });

  test("propagates non-1.0 derivative correctly", () => {
    const x = new Scalar(3);
    const y = new Scalar(4);
    const z = x.mul(y);
    
    backPropagate(z, 2.0);  // Start with derivative of 2
    
    // dz/dx = y * 2 = 8
    expect(x.derivative).toBeCloseTo(8, 5);
    // dz/dy = x * 2 = 6
    expect(y.derivative).toBeCloseTo(6, 5);
  });
});

// ============================================================
// Gradient Comparison with Central Difference (task1_4)
// ============================================================

describe("Gradient correctness via central difference", () => {
  test.prop([positiveFloat])("log gradient matches numerical", (xVal) => {
    const x = new Scalar(xVal);
    const z = x.log();
    backPropagate(z, 1.0);
    
    const numerical = centralDifference((a) => Math.log(a), [xVal], 0);
    assertClose(x.derivative!, numerical, 4);
  });

  test.prop([fc.double({ noNaN: true, min: -5, max: 5 })])("exp gradient matches numerical", (xVal) => {
    const x = new Scalar(xVal);
    const z = x.exp();
    backPropagate(z, 1.0);
    
    const numerical = centralDifference((a) => Math.exp(a), [xVal], 0);
    assertClose(x.derivative!, numerical, 4);
  });

  test.prop([smallFloat])("sigmoid gradient matches numerical", (xVal) => {
    const sigmoid = (a: number) => 1 / (1 + Math.exp(-a));
    
    const x = new Scalar(xVal);
    const z = x.sigmoid();
    backPropagate(z, 1.0);
    
    const numerical = centralDifference(sigmoid, [xVal], 0);
    assertClose(x.derivative!, numerical, 4);
  });

  test.prop([smallFloat, smallFloat])("mul gradient matches numerical", (xVal, yVal) => {
    const x = new Scalar(xVal);
    const y = new Scalar(yVal);
    const z = x.mul(y);
    backPropagate(z, 1.0);
    
    const fn = (a: number, b: number) => a * b;
    assertClose(x.derivative!, centralDifference(fn, [xVal, yVal], 0), 4);
    assertClose(y.derivative!, centralDifference(fn, [xVal, yVal], 1), 4);
  });

  test.prop([smallFloat, smallFloat])("add gradient matches numerical", (xVal, yVal) => {
    const x = new Scalar(xVal);
    const y = new Scalar(yVal);
    const z = x.add(y);
    backPropagate(z, 1.0);
    
    const fn = (a: number, b: number) => a + b;
    assertClose(x.derivative!, centralDifference(fn, [xVal, yVal], 0), 4);
    assertClose(y.derivative!, centralDifference(fn, [xVal, yVal], 1), 4);
  });
});

// ============================================================
// Task 2.4 - Tensor Autograd Tests
// ============================================================

import { Tensor } from "./tensor.js";
import { topologicalSortTensor, backPropagateTensor } from "./autodiff.js";

/**
 * Numerically check gradient of a tensor function
 * Similar to minitorch's grad_check
 */
function tensorGradCheck(
  fn: (...tensors: Tensor[]) => Tensor,
  ...inputs: Tensor[]
): void {
  const epsilon = 1e-5;
  
  // Compute analytical gradients via backward
  const output = fn(...inputs);
  const scalarOutput = output.sum();
  scalarOutput.backward();

  // Check each input
  for (let inputIdx = 0; inputIdx < inputs.length; inputIdx++) {
    const input = inputs[inputIdx]!;
    const grad = input.grad;
    expect(grad).not.toBeNull();
    expect(grad!.shape).toEqual(input.shape);

    // Check each element's gradient numerically
    for (let i = 0; i < input.size; i++) {
      // Convert flat index to multi-index
      const idx: number[] = [];
      let remaining = i;
      for (let d = input.dims - 1; d >= 0; d--) {
        idx.unshift(remaining % input.shape[d]!);
        remaining = Math.floor(remaining / input.shape[d]!);
      }

      // Compute numerical gradient
      const originalVal = input.get(idx);
      
      input.set(idx, originalVal + epsilon);
      const plusOutput = fn(...inputs).sum().item();
      
      input.set(idx, originalVal - epsilon);
      const minusOutput = fn(...inputs).sum().item();
      
      input.set(idx, originalVal); // Restore
      
      const numericalGrad = (plusOutput - minusOutput) / (2 * epsilon);
      const analyticalGrad = grad!.get(idx);
      
      expect(analyticalGrad).toBeCloseTo(numericalGrad, 3);
    }
    
    // Reset grad for next check
    input.zero_grad_();
  }
}

describe("Tensor Autograd - Task 2.4", () => {
  describe("topologicalSortTensor", () => {
    test("sorts single tensor (leaf)", () => {
      const a = Tensor.tensor([1, 2, 3]);
      const sorted = topologicalSortTensor(a);
      expect(sorted).toEqual([a]);
    });

    test("sorts simple chain", () => {
      const a = Tensor.tensor([1, 2, 3]);
      const b = a.neg();
      const c = b.sigmoid();
      const sorted = topologicalSortTensor(c);
      expect(sorted.length).toBe(3);
      expect(sorted[0]).toBe(c);
      expect(sorted[sorted.length - 1]).toBe(a);
    });

    test("sorts diamond pattern", () => {
      const a = Tensor.tensor([1, 2, 3]);
      const b = a.neg();
      const c = a.exp();
      const d = b.add(c);
      const sorted = topologicalSortTensor(d);
      expect(sorted.length).toBe(4);
      expect(sorted[0]).toBe(d);
      // 'a' should come last (deepest ancestor)
      expect(sorted[sorted.length - 1]).toBe(a);
    });
  });

  describe("One-arg backward functions", () => {
    test("neg backward", () => {
      const a = Tensor.tensor([1, 2, 3, 4]);
      const b = a.neg();
      b.sum().backward();
      
      expect(a.grad).not.toBeNull();
      expect(a.grad!.shape).toEqual(a.shape);
      // d(-x)/dx = -1
      expect(a.grad!.toArray()).toEqual([-1, -1, -1, -1]);
    });

    test("neg backward with grad_check", () => {
      const a = Tensor.tensor([1, -2, 0.5, 3]);
      tensorGradCheck((t) => t.neg(), a);
    });

    test("sigmoid backward", () => {
      const a = Tensor.tensor([0, 1, -1]);
      const b = a.sigmoid();
      b.sum().backward();
      
      expect(a.grad).not.toBeNull();
      // d(sigmoid)/dx = sigmoid * (1 - sigmoid)
      for (let i = 0; i < a.size; i++) {
        const sig = 1 / (1 + Math.exp(-a.get([i])));
        const expected = sig * (1 - sig);
        expect(a.grad!.get([i])).toBeCloseTo(expected, 5);
      }
    });

    test("sigmoid backward with grad_check", () => {
      const a = Tensor.tensor([0.5, -0.5, 1, -1]);
      tensorGradCheck((t) => t.sigmoid(), a);
    });

    test("relu backward", () => {
      const a = Tensor.tensor([-2, -1, 0, 1, 2]);
      const b = a.relu();
      b.sum().backward();
      
      expect(a.grad).not.toBeNull();
      // d(relu)/dx = 1 if x > 0 else 0
      expect(a.grad!.toArray()).toEqual([0, 0, 0, 1, 1]);
    });

    test("relu backward with grad_check", () => {
      const a = Tensor.tensor([0.5, -0.5, 2, -2]);
      tensorGradCheck((t) => t.relu(), a);
    });

    test("log backward", () => {
      const a = Tensor.tensor([1, 2, Math.E]);
      const b = a.log();
      b.sum().backward();
      
      expect(a.grad).not.toBeNull();
      // d(log x)/dx = 1/x
      expect(a.grad!.get([0])).toBeCloseTo(1, 5);
      expect(a.grad!.get([1])).toBeCloseTo(0.5, 5);
      expect(a.grad!.get([2])).toBeCloseTo(1/Math.E, 5);
    });

    test("log backward with grad_check", () => {
      const a = Tensor.tensor([0.5, 1, 2, 5]);
      tensorGradCheck((t) => t.log(), a);
    });

    test("exp backward", () => {
      const a = Tensor.tensor([0, 1, 2]);
      const b = a.exp();
      b.sum().backward();
      
      expect(a.grad).not.toBeNull();
      // d(exp x)/dx = exp(x)
      expect(a.grad!.get([0])).toBeCloseTo(1, 5);
      expect(a.grad!.get([1])).toBeCloseTo(Math.E, 5);
      expect(a.grad!.get([2])).toBeCloseTo(Math.exp(2), 5);
    });

    test("exp backward with grad_check", () => {
      const a = Tensor.tensor([-1, 0, 1, 2]);
      tensorGradCheck((t) => t.exp(), a);
    });

    test("inv backward", () => {
      const a = Tensor.tensor([1, 2, 4]);
      const b = a.inv();
      b.sum().backward();
      
      expect(a.grad).not.toBeNull();
      // d(1/x)/dx = -1/x^2
      expect(a.grad!.get([0])).toBeCloseTo(-1, 5);
      expect(a.grad!.get([1])).toBeCloseTo(-0.25, 5);
      expect(a.grad!.get([2])).toBeCloseTo(-1/16, 5);
    });

    test("inv backward with grad_check", () => {
      const a = Tensor.tensor([0.5, 1, 2, 4]);
      tensorGradCheck((t) => t.inv(), a);
    });
  });

  describe("Two-arg backward functions", () => {
    test("add backward - same shapes", () => {
      const a = Tensor.tensor([1, 2, 3]);
      const b = Tensor.tensor([4, 5, 6]);
      const c = a.add(b);
      c.sum().backward();
      
      expect(a.grad).not.toBeNull();
      expect(b.grad).not.toBeNull();
      // d(a+b)/da = 1, d(a+b)/db = 1
      expect(a.grad!.toArray()).toEqual([1, 1, 1]);
      expect(b.grad!.toArray()).toEqual([1, 1, 1]);
    });

    test("add backward with grad_check", () => {
      const a = Tensor.tensor([1, 2, 3]);
      const b = Tensor.tensor([4, 5, 6]);
      tensorGradCheck((x, y) => x.add(y), a, b);
    });

    test("mul backward - same shapes", () => {
      const a = Tensor.tensor([1, 2, 3]);
      const b = Tensor.tensor([4, 5, 6]);
      const c = a.mul(b);
      c.sum().backward();
      
      expect(a.grad).not.toBeNull();
      expect(b.grad).not.toBeNull();
      // d(a*b)/da = b, d(a*b)/db = a
      expect(a.grad!.toArray()).toEqual([4, 5, 6]);
      expect(b.grad!.toArray()).toEqual([1, 2, 3]);
    });

    test("mul backward with grad_check", () => {
      const a = Tensor.tensor([1, 2, 3]);
      const b = Tensor.tensor([4, 5, 6]);
      tensorGradCheck((x, y) => x.mul(y), a, b);
    });

    test("sub backward", () => {
      const a = Tensor.tensor([5, 6, 7]);
      const b = Tensor.tensor([1, 2, 3]);
      const c = a.sub(b);
      c.sum().backward();
      
      expect(a.grad).not.toBeNull();
      expect(b.grad).not.toBeNull();
      // d(a-b)/da = 1, d(a-b)/db = -1
      expect(a.grad!.toArray()).toEqual([1, 1, 1]);
      expect(b.grad!.toArray()).toEqual([-1, -1, -1]);
    });
  });

  describe("Broadcast backward", () => {
    test("add with scalar broadcast - gradient reduces correctly", () => {
      const a = Tensor.tensor([[1, 2], [3, 4]]);  // shape [2, 2]
      const b = Tensor.tensor([10]);              // shape [1]
      const c = a.add(b);
      c.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 2]);
      expect(b.grad!.shape).toEqual([1]);
      // b was broadcast 4 times, so its gradient should be summed
      expect(b.grad!.get([0])).toBe(4);
    });

    test("add with row broadcast", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);  // shape [2, 3]
      const b = Tensor.tensor([10, 20, 30]);            // shape [3]
      const c = a.add(b);
      c.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 3]);
      expect(b.grad!.shape).toEqual([3]);
      // b was broadcast along dim 0, so gradient sums along dim 0
      expect(b.grad!.toArray()).toEqual([2, 2, 2]);
    });

    test("mul with broadcast - gradient shapes match inputs", () => {
      const a = Tensor.tensor([1]);           // shape [1]
      const b = Tensor.tensor([[1, 2]]);      // shape [1, 2]
      const c = a.mul(b);
      c.sum().backward();
      
      expect(a.grad!.shape).toEqual(a.shape);
      expect(b.grad!.shape).toEqual(b.shape);
    });

    test("mul with broadcast grad_check", () => {
      const a = Tensor.tensor([2]);
      const b = Tensor.tensor([[1, 2, 3]]);
      tensorGradCheck((x, y) => x.mul(y), a, b);
    });

    test("complex broadcast grad_check", () => {
      const a = Tensor.tensor([[1, 2], [3, 4]]);  // [2, 2]
      const b = Tensor.tensor([5, 6]);            // [2]
      tensorGradCheck((x, y) => x.mul(y).add(y), a, b);
    });

    test("lt backward with broadcast - gradient shapes match inputs", () => {
      const a = Tensor.tensor([[1, 2], [3, 4]]);  // [2, 2]
      const b = Tensor.tensor([2.5]);             // [1]
      const c = a.lt(b).sum();
      c.backward();
      
      expect(a.grad!.shape).toEqual([2, 2]);
      expect(b.grad!.shape).toEqual([1]);
      // LT has zero gradient everywhere
      expect(a.grad!.toArray()).toEqual([[0, 0], [0, 0]]);
      expect(b.grad!.toArray()).toEqual([0]);
    });

    test("eq backward with broadcast - gradient shapes match inputs", () => {
      const a = Tensor.tensor([[1, 2], [3, 4]]);  // [2, 2]
      const b = Tensor.tensor([2]);               // [1]
      const c = a.eq(b).sum();
      c.backward();
      
      expect(a.grad!.shape).toEqual([2, 2]);
      expect(b.grad!.shape).toEqual([1]);
      // EQ has zero gradient everywhere
      expect(a.grad!.toArray()).toEqual([[0, 0], [0, 0]]);
      expect(b.grad!.toArray()).toEqual([0]);
    });
  });

  describe("Reduce backward", () => {
    test("sum along dim backward", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);  // shape [2, 3]
      const b = a.sum(1);  // shape [2, 1]
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 3]);
      // All gradients should be 1 (sum just passes gradient through)
      expect(a.grad!.toArray()).toEqual([[1, 1, 1], [1, 1, 1]]);
    });

    test("sum all backward", () => {
      const a = Tensor.tensor([[1, 2], [3, 4]]);
      const b = a.sum();
      b.backward();
      
      expect(a.grad!.shape).toEqual([2, 2]);
      expect(a.grad!.toArray()).toEqual([[1, 1], [1, 1]]);
    });

    test("mean backward", () => {
      const a = Tensor.tensor([1, 2, 3, 4]);
      const b = a.mean();
      b.backward();
      
      expect(a.grad!.shape).toEqual([4]);
      // d(mean)/dx_i = 1/n
      for (let i = 0; i < 4; i++) {
        expect(a.grad!.get([i])).toBeCloseTo(0.25, 5);
      }
    });
  });

  describe("Permute backward", () => {
    test("permute 2D backward", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);  // shape [2, 3]
      const b = a.permute(1, 0);  // shape [3, 2]
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 3]);
      expect(a.grad!.toArray()).toEqual([[1, 1, 1], [1, 1, 1]]);
    });

    test("permute grad_check", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
      tensorGradCheck((t) => t.permute(1, 0), a);
    });

    test("permute 3D backward", () => {
      const a = Tensor.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);  // shape [2, 2, 2]
      const b = a.permute(2, 0, 1);  // shape [2, 2, 2]
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 2, 2]);
    });
  });

  describe("View backward", () => {
    test("view backward restores shape", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);  // shape [2, 3]
      const b = a.view(6);  // shape [6]
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 3]);
      expect(a.grad!.toArray()).toEqual([[1, 1, 1], [1, 1, 1]]);
    });

    test("view grad_check", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
      tensorGradCheck((t) => t.contiguous().view(6), a);
    });

    test("view reshape grad_check", () => {
      const a = Tensor.tensor([1, 2, 3, 4, 5, 6]);
      tensorGradCheck((t) => t.view(2, 3), a);
    });

    test("contiguous grad_check", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
      tensorGradCheck((t) => t.contiguous(), a);
    });
  });

  describe("Complex computation graphs", () => {
    test("chain of operations", () => {
      const a = Tensor.tensor([1, 2, 3, 4]);
      const b = a.sigmoid().relu().log();
      b.sum().backward();
      
      expect(a.grad).not.toBeNull();
      expect(a.grad!.shape).toEqual([4]);
    });

    test("diamond pattern - same tensor used twice", () => {
      const a = Tensor.tensor([1, 2, 3]);
      const b = a.mul(a);  // a^2
      b.sum().backward();
      
      expect(a.grad).not.toBeNull();
      // d(a^2)/da = 2a
      expect(a.grad!.toArray()).toEqual([2, 4, 6]);
    });

    test("complex expression grad_check", () => {
      const a = Tensor.tensor([0.5, 1, 1.5, 2]);
      tensorGradCheck((t) => t.sigmoid().mul(t).add(t.exp()), a);
    });

    test("multiple inputs complex expression", () => {
      const a = Tensor.tensor([1, 2, 3]);
      const b = Tensor.tensor([0.5, 0.5, 0.5]);
      tensorGradCheck((x, y) => x.mul(y).add(x).sigmoid(), a, b);
    });
  });

  describe("Gradient accumulation", () => {
    test("gradients accumulate correctly", () => {
      const a = Tensor.tensor([1, 2, 3]);
      const b = a.add(a);  // 2*a
      b.sum().backward();
      
      // Both uses of 'a' contribute gradients
      expect(a.grad!.toArray()).toEqual([2, 2, 2]);
    });

    test("zero_grad_ resets gradients", () => {
      const a = Tensor.tensor([1, 2, 3]);
      a.neg().sum().backward();
      expect(a.grad).not.toBeNull();
      
      a.zero_grad_();
      expect(a.grad).toBeNull();
    });

    test("multiple backward calls accumulate", () => {
      const a = Tensor.tensor([1, 2, 3]);
      a.neg().sum().backward();
      const firstGrad = a.grad!.toArray();
      
      // Don't zero_grad, do another backward
      a.neg().sum().backward();
      
      // Gradients should have doubled
      expect(a.grad!.get([0])).toBe(firstGrad[0]! * 2);
    });
  });

  describe("Edge cases", () => {
    test("scalar tensor backward", () => {
      const a = Tensor.tensor(5);
      const b = a.mul(2);
      b.backward();
      
      expect(a.grad!.get([])).toBe(2);
    });

    test("1x1 matrix backward", () => {
      const a = Tensor.tensor([[3]]);
      const b = a.mul(a);  // a^2 = 9
      b.sum().backward();
      
      expect(a.grad!.get([0, 0])).toBe(6);  // 2*a = 6
    });

    test("large tensor backward", () => {
      const a = Tensor.rand([10, 10]);
      const b = a.sigmoid().sum();
      b.backward();
      
      expect(a.grad!.shape).toEqual([10, 10]);
    });
  });

  describe("Minitorch test_grad_size equivalent", () => {
    test("gradient shapes match input shapes after broadcast", () => {
      const a = Tensor.tensor([1]);           // shape [1]
      const b = Tensor.tensor([[1, 1]]);      // shape [1, 2]
      const c = a.mul(b).sum();
      
      c.backward();
      
      expect(c.shape).toEqual([]);
      expect(a.grad).not.toBeNull();
      expect(b.grad).not.toBeNull();
      expect(a.grad!.shape).toEqual(a.shape);
      expect(b.grad!.shape).toEqual(b.shape);
    });
  });

  describe("Unbroadcast edge cases", () => {
    test("unbroadcast to scalar - add", () => {
      const a = Tensor.tensor([[1, 2], [3, 4]]);  // shape [2, 2]
      const b = Tensor.tensor(5);                 // scalar, shape []
      const c = a.add(b).sum();
      c.backward();
      
      expect(a.grad!.shape).toEqual([2, 2]);
      expect(b.grad!.shape).toEqual([]);
      // b was broadcast to all 4 elements, so gradient should be sum = 4
      expect(b.grad!.get([])).toBe(4);
    });

    test("unbroadcast to scalar - mul", () => {
      const a = Tensor.tensor([1, 2, 3]);  // shape [3]
      const b = Tensor.tensor(2);          // scalar
      const c = a.mul(b).sum();
      c.backward();
      
      expect(b.grad!.shape).toEqual([]);
      // d(sum(a*b))/db = sum(a) = 1+2+3 = 6
      expect(b.grad!.get([])).toBe(6);
    });

    test("unbroadcast from 3D to 1D", () => {
      const a = Tensor.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);  // shape [2, 2, 2]
      const b = Tensor.tensor([10, 20]);  // shape [2]
      const c = a.add(b).sum();
      c.backward();
      
      expect(a.grad!.shape).toEqual([2, 2, 2]);
      expect(b.grad!.shape).toEqual([2]);
      // b is broadcast along dims 0 and 1, so gradient sums along those dims
      // Each element of b appears 4 times (2*2)
      expect(b.grad!.toArray()).toEqual([4, 4]);
    });

    test("unbroadcast from 3D to scalar", () => {
      const a = Tensor.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);  // shape [2, 2, 2]
      const b = Tensor.tensor(1);  // scalar
      const c = a.mul(b).sum();
      c.backward();
      
      expect(b.grad!.shape).toEqual([]);
      // d(sum(a*b))/db = sum(a) = 1+2+3+4+5+6+7+8 = 36
      expect(b.grad!.get([])).toBe(36);
    });

    test("unbroadcast with size-1 dimensions", () => {
      const a = Tensor.tensor([[1, 2, 3]]);       // shape [1, 3]
      const b = Tensor.tensor([[10], [20]]);     // shape [2, 1]
      const c = a.add(b).sum();
      c.backward();
      
      expect(a.grad!.shape).toEqual([1, 3]);
      expect(b.grad!.shape).toEqual([2, 1]);
      // a is broadcast along dim 0, so gradient sums to [2, 2, 2]
      expect(a.grad!.toArray()).toEqual([[2, 2, 2]]);
      // b is broadcast along dim 1, so gradient sums to [[3], [3]]
      expect(b.grad!.toArray()).toEqual([[3], [3]]);
    });

    test("unbroadcast mul grad_check with scalar", () => {
      const a = Tensor.tensor([[1, 2], [3, 4]]);
      const b = Tensor.tensor(2);
      tensorGradCheck((x, y) => x.mul(y), a, b);
    });

    test("unbroadcast add grad_check from 3D to 1D", () => {
      const a = Tensor.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
      const b = Tensor.tensor([1, 2]);
      tensorGradCheck((x, y) => x.add(y), a, b);
    });
  });

  describe("Contiguous edge cases", () => {
    test("contiguous on already contiguous tensor", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
      const b = a.contiguous();
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 3]);
      expect(a.grad!.toArray()).toEqual([[1, 1, 1], [1, 1, 1]]);
    });

    test("contiguous after permute", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);  // shape [2, 3]
      const b = a.permute(1, 0);  // shape [3, 2], non-contiguous
      const c = b.contiguous();   // now contiguous
      c.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 3]);
      expect(a.grad!.toArray()).toEqual([[1, 1, 1], [1, 1, 1]]);
    });

    test("contiguous grad_check after permute", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
      tensorGradCheck((t) => t.permute(1, 0).contiguous(), a);
    });

    test("multiple contiguous calls", () => {
      const a = Tensor.tensor([1, 2, 3, 4]);
      const b = a.contiguous().contiguous().contiguous();
      b.sum().backward();
      
      expect(a.grad!.toArray()).toEqual([1, 1, 1, 1]);
    });

    test("contiguous on scalar", () => {
      const a = Tensor.tensor(5);
      const b = a.contiguous();
      b.backward();
      
      expect(a.grad!.get([])).toBe(1);
    });
  });

  describe("Sum edge cases", () => {
    test("sum of scalar tensor", () => {
      const a = Tensor.tensor(5);
      const b = a.sum();
      b.backward();
      
      expect(a.grad!.shape).toEqual([]);
      expect(a.grad!.get([])).toBe(1);
    });

    test("sum of 1-element tensor", () => {
      const a = Tensor.tensor([42]);
      const b = a.sum();
      b.backward();
      
      expect(a.grad!.shape).toEqual([1]);
      expect(a.grad!.get([0])).toBe(1);
    });

    test("sum all dims of 3D tensor", () => {
      const a = Tensor.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);  // shape [2, 2, 2]
      const b = a.sum();
      b.backward();
      
      expect(a.grad!.shape).toEqual([2, 2, 2]);
      // All gradients should be 1
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          for (let k = 0; k < 2; k++) {
            expect(a.grad!.get([i, j, k])).toBe(1);
          }
        }
      }
    });

    test("sum along first dim", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);  // shape [2, 3]
      const b = a.sum(0);  // shape [1, 3]
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 3]);
      expect(a.grad!.toArray()).toEqual([[1, 1, 1], [1, 1, 1]]);
    });

    test("sum along last dim", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);  // shape [2, 3]
      const b = a.sum(1);  // shape [2, 1]
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 3]);
      expect(a.grad!.toArray()).toEqual([[1, 1, 1], [1, 1, 1]]);
    });

    test("sum along middle dim of 3D", () => {
      const a = Tensor.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);  // shape [2, 2, 2]
      const b = a.sum(1);  // shape [2, 1, 2]
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 2, 2]);
    });

    test("sum grad_check - all dims", () => {
      const a = Tensor.tensor([[1, 2], [3, 4]]);
      tensorGradCheck((t) => t.sum(), a);
    });

    test("sum grad_check - specific dim", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
      tensorGradCheck((t) => t.sum(0), a);
      a.zero_grad_();
      tensorGradCheck((t) => t.sum(1), a);
    });

    test("chained sum operations", () => {
      const a = Tensor.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);  // shape [2, 2, 2]
      const b = a.sum(2).sum(1).sum(0);  // reduce all dims one by one
      b.backward();
      
      expect(a.grad!.shape).toEqual([2, 2, 2]);
    });
  });

  describe("All edge cases (forward only - no gradient)", () => {
    test("all of tensor with all ones", () => {
      const a = Tensor.tensor([[1, 1], [1, 1]]);
      const b = a.all();
      expect(b.item()).toBe(1);
    });

    test("all of tensor with a zero", () => {
      const a = Tensor.tensor([[1, 1], [0, 1]]);
      const b = a.all();
      expect(b.item()).toBe(0);
    });

    test("all along dim 0", () => {
      const a = Tensor.tensor([[1, 0], [1, 1]]);
      const b = a.all(0);
      expect(b.toArray()).toEqual([[1, 0]]);
    });

    test("all along dim 1", () => {
      const a = Tensor.tensor([[1, 0], [1, 1]]);
      const b = a.all(1);
      expect(b.toArray()).toEqual([[0], [1]]);
    });

    test("all of scalar", () => {
      const a = Tensor.tensor(1);
      const b = a.all();
      expect(b.item()).toBe(1);
    });

    test("all of scalar zero", () => {
      const a = Tensor.tensor(0);
      const b = a.all();
      expect(b.item()).toBe(0);
    });

    test("all of 3D tensor", () => {
      const a = Tensor.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]);
      const b = a.all();
      expect(b.item()).toBe(1);
    });

    test("all of 3D tensor with zero", () => {
      const a = Tensor.tensor([[[1, 1], [1, 1]], [[1, 0], [1, 1]]]);
      const b = a.all();
      expect(b.item()).toBe(0);
    });
  });

  describe("Mean edge cases", () => {
    test("mean of scalar", () => {
      const a = Tensor.tensor(10);
      const b = a.mean();
      b.backward();
      
      expect(a.grad!.get([])).toBeCloseTo(1, 5);
    });

    test("mean of 1-element tensor", () => {
      const a = Tensor.tensor([7]);
      const b = a.mean();
      b.backward();
      
      expect(a.grad!.get([0])).toBeCloseTo(1, 5);
    });

    test("mean along dim 0", () => {
      const a = Tensor.tensor([[2, 4], [6, 8]]);  // shape [2, 2]
      const b = a.mean(0);  // shape [1, 2], values [4, 6]
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 2]);
      // d(mean)/dx = 1/n where n=2 for each element
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          expect(a.grad!.get([i, j])).toBeCloseTo(0.5, 5);
        }
      }
    });

    test("mean along dim 1", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);  // shape [2, 3]
      const b = a.mean(1);
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 3]);
      // d(mean)/dx = 1/n where n=3 for each element
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 3; j++) {
          expect(a.grad!.get([i, j])).toBeCloseTo(1/3, 5);
        }
      }
    });

    test("mean grad_check - all elements", () => {
      const a = Tensor.tensor([[1, 2], [3, 4]]);
      tensorGradCheck((t) => t.mean(), a);
    });

    test("mean grad_check - specific dim", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
      tensorGradCheck((t) => t.mean(0), a);
      a.zero_grad_();
      tensorGradCheck((t) => t.mean(1), a);
    });
  });

  describe("View edge cases", () => {
    test("view to same shape", () => {
      const a = Tensor.tensor([[1, 2], [3, 4]]);
      const b = a.view(2, 2);
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 2]);
      expect(a.grad!.toArray()).toEqual([[1, 1], [1, 1]]);
    });

    test("view to 1D", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
      const b = a.view(6);
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 3]);
    });

    test("view from 1D to 2D", () => {
      const a = Tensor.tensor([1, 2, 3, 4, 5, 6]);
      const b = a.view(2, 3);
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([6]);
    });

    test("view to 3D", () => {
      const a = Tensor.tensor([1, 2, 3, 4, 5, 6, 7, 8]);
      const b = a.view(2, 2, 2);
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([8]);
    });

    test("view to scalar-like shape", () => {
      const a = Tensor.tensor([5]);
      const b = a.view();  // scalar shape []
      b.backward();
      
      expect(a.grad!.shape).toEqual([1]);
      expect(a.grad!.get([0])).toBe(1);
    });

    test("view grad_check with reshape", () => {
      const a = Tensor.tensor([1, 2, 3, 4, 5, 6]);
      tensorGradCheck((t) => t.view(3, 2), a);
    });

    test("chained view operations", () => {
      const a = Tensor.tensor([1, 2, 3, 4, 5, 6]);
      const b = a.view(2, 3).view(3, 2).view(6);
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([6]);
    });
  });

  describe("Permute edge cases", () => {
    test("permute identity (no change)", () => {
      const a = Tensor.tensor([[1, 2], [3, 4]]);
      const b = a.permute(0, 1);  // identity permutation
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 2]);
    });

    test("permute 3D - reverse order", () => {
      const a = Tensor.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);  // shape [2, 2, 2]
      const b = a.permute(2, 1, 0);  // reverse dims
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 2, 2]);
    });

    test("permute and permute back", () => {
      const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);  // shape [2, 3]
      const b = a.permute(1, 0).permute(1, 0);  // should be back to original
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 3]);
    });

    test("permute 4D tensor", () => {
      const a = Tensor.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 
                               [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]);  // shape [2, 2, 2, 2]
      const b = a.permute(3, 2, 1, 0);
      b.sum().backward();
      
      expect(a.grad!.shape).toEqual([2, 2, 2, 2]);
    });

    test("permute grad_check 3D", () => {
      const a = Tensor.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
      tensorGradCheck((t) => t.permute(2, 0, 1), a);
    });
  });
});
