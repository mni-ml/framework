// fc generates suites of inputs
import { test, fc } from '@fast-check/jest';
import { describe, expect } from '@jest/globals';
import { Scalar } from "./scalar.js";

const DIGIT_TOLERANCE = 5;

/** floats from set of small finite floats to avoid inputs that are false failures like NaN/Infinity, extreme over/underflow. */
const smallFloat = fc.double({ noNaN: true, min: -1000, max: 1000 });

/** Positive floats for operations like log that need positive inputs */
const positiveFloat = fc.double({ noNaN: true, min: 0.001, max: 1000 });

/** Non-zero floats for division */
const nonZeroFloat = fc.double({ noNaN: true, min: -1000, max: 1000 }).filter(x => Math.abs(x) > 1e-5);

/** A looser float comparator for nonlinear ops (sigmoid etc.). */
function assertClose(actual: number, expected: number, digits = DIGIT_TOLERANCE) {
  expect(actual).toBeCloseTo(expected, digits);
}

// ============================================================
// Task 1.2 - Scalar Basic Operations
// ============================================================

describe("Scalar construction", () => {
  test("Scalar stores value correctly", () => {
    fc.assert(
      fc.property(smallFloat, (a) => {
        const s = new Scalar(a);
        expect(s.data).toBe(a);
      })
    );
  });

  test("Each Scalar has unique id", () => {
    const s1 = new Scalar(1);
    const s2 = new Scalar(2);
    const s3 = new Scalar(3);
    expect(s1.uniqueId).not.toBe(s2.uniqueId);
    expect(s2.uniqueId).not.toBe(s3.uniqueId);
  });

  test("Scalar toString returns expected format", () => {
    const s = new Scalar(42);
    expect(s.toString()).toBe("Scalar(42)");
  });
});

describe("Scalar arithmetic operations", () => {
  test("add returns correct value", () => {
    fc.assert(
      fc.property(smallFloat, smallFloat, (a, b) => {
        const x = new Scalar(a);
        const y = new Scalar(b);
        const z = x.add(y);
        assertClose(z.data, a + b);
      })
    );
  });

  test("add works with raw numbers (ScalarLike)", () => {
    fc.assert(
      fc.property(smallFloat, smallFloat, (a, b) => {
        const x = new Scalar(a);
        const z = x.add(b);  // b is a raw number
        assertClose(z.data, a + b);
      })
    );
  });

  test("mul returns correct value", () => {
    fc.assert(
      fc.property(smallFloat, smallFloat, (a, b) => {
        const x = new Scalar(a);
        const y = new Scalar(b);
        const z = x.mul(y);
        assertClose(z.data, a * b);
      })
    );
  });

  test("mul works with raw numbers (ScalarLike)", () => {
    fc.assert(
      fc.property(smallFloat, smallFloat, (a, b) => {
        const x = new Scalar(a);
        const z = x.mul(b);
        assertClose(z.data, a * b);
      })
    );
  });

  test("sub returns correct value", () => {
    fc.assert(
      fc.property(smallFloat, smallFloat, (a, b) => {
        const x = new Scalar(a);
        const y = new Scalar(b);
        const z = x.sub(y);
        assertClose(z.data, a - b);
      })
    );
  });

  test("neg returns correct value", () => {
    fc.assert(
      fc.property(smallFloat, (a) => {
        const x = new Scalar(a);
        const z = x.neg();
        expect(z.data).toBe(-a);
      })
    );
  });

  test("div returns correct value", () => {
    fc.assert(
      fc.property(smallFloat, nonZeroFloat, (a, b) => {
        const x = new Scalar(a);
        const y = new Scalar(b);
        const z = x.div(y);
        assertClose(z.data, a / b);
      })
    );
  });
});

describe("Scalar comparison operations", () => {
  test("lt returns 1.0 when less than, 0.0 otherwise", () => {
    fc.assert(
      fc.property(smallFloat, (a) => {
        const x = new Scalar(a);
        expect(x.lt(a + 1).data).toBe(1.0);  // a < a+1
        expect(x.lt(a - 1).data).toBe(0.0);  // a < a-1 is false
        expect(x.lt(a).data).toBe(0.0);      // a < a is false
      })
    );
  });

  test("gt returns 1.0 when greater than, 0.0 otherwise", () => {
    fc.assert(
      fc.property(smallFloat, (a) => {
        const x = new Scalar(a);
        expect(x.gt(a - 1).data).toBe(1.0);  // a > a-1
        expect(x.gt(a + 1).data).toBe(0.0);  // a > a+1 is false
        expect(x.gt(a).data).toBe(0.0);      // a > a is false
      })
    );
  });

  test("eq returns 1.0 when equal, 0.0 otherwise", () => {
    fc.assert(
      fc.property(smallFloat, (a) => {
        const x = new Scalar(a);
        expect(x.eq(a).data).toBe(1.0);
        expect(x.eq(a + 1).data).toBe(0.0);
        expect(x.eq(a - 1).data).toBe(0.0);
      })
    );
  });
});

describe("Scalar mathematical functions", () => {
  test("log returns correct value", () => {
    fc.assert(
      fc.property(positiveFloat, (a) => {
        const x = new Scalar(a);
        const z = x.log();
        assertClose(z.data, Math.log(a), 4);  // Looser tolerance due to EPS in operators.log
      })
    );
  });

  test("exp returns correct value", () => {
    // Use smaller range to avoid overflow
    const boundedFloat = fc.double({ noNaN: true, min: -100, max: 100 });
    fc.assert(
      fc.property(boundedFloat, (a) => {
        const x = new Scalar(a);
        const z = x.exp();
        assertClose(z.data, Math.exp(a));
      })
    );
  });

  test("sigmoid returns value between 0 and 1", () => {
    fc.assert(
      fc.property(smallFloat, (a) => {
        const x = new Scalar(a);
        const z = x.sigmoid();
        expect(z.data).toBeGreaterThanOrEqual(0.0);
        expect(z.data).toBeLessThanOrEqual(1.0);
      })
    );
  });

  test("sigmoid(0) equals 0.5", () => {
    const x = new Scalar(0);
    assertClose(x.sigmoid().data, 0.5, 10);
  });

  test("relu returns max(0, x)", () => {
    fc.assert(
      fc.property(smallFloat, (a) => {
        const x = new Scalar(a);
        const z = x.relu();
        if (a > 0) {
          expect(z.data).toBe(a);
        } else {
          expect(z.data).toBe(0);
        }
      })
    );
  });
});

// ============================================================
// Task 1.2 - Scalar History Tracking
// ============================================================

describe("Scalar history tracking", () => {
  test("operations record history", () => {
    const x = new Scalar(2);
    const y = new Scalar(3);
    const z = x.add(y);

    expect(z.history).not.toBeNull();
    expect(z.history!.inputs.length).toBe(2);
    expect(z.history!.ctx).not.toBeNull();
    expect(z.history!.lastFn).not.toBeNull();
  });

  test("history inputs contain correct values", () => {
    const x = new Scalar(2);
    const y = new Scalar(3);
    const z = x.add(y);

    expect(z.history!.inputs[0]!.data).toBe(2);
    expect(z.history!.inputs[1]!.data).toBe(3);
  });

  test("chained operations build computation graph", () => {
    const x = new Scalar(2);
    const y = new Scalar(3);
    const z = x.add(y).mul(y);  // (2 + 3) * 3 = 15

    expect(z.data).toBe(15);
    expect(z.history).not.toBeNull();
    
    // z's history should reference the intermediate add result
    const mulInputs = z.history!.inputs;
    expect(mulInputs.length).toBe(2);
    expect(mulInputs[0]!.data).toBe(5);  // Result of x.add(y)
    expect(mulInputs[1]!.data).toBe(3);  // y
  });
});

// ============================================================
// Task 1.2 - Property Tests (Algebraic Laws)
// ============================================================

describe("Scalar algebraic properties", () => {
  test("addition is commutative: x + y == y + x", () => {
    fc.assert(
      fc.property(smallFloat, smallFloat, (a, b) => {
        const x = new Scalar(a);
        const y = new Scalar(b);
        assertClose(x.add(y).data, y.add(x).data);
      })
    );
  });

  test("multiplication is commutative: x * y == y * x", () => {
    fc.assert(
      fc.property(smallFloat, smallFloat, (a, b) => {
        const x = new Scalar(a);
        const y = new Scalar(b);
        assertClose(x.mul(y).data, y.mul(x).data);
      })
    );
  });

  test("double negation: neg(neg(x)) == x", () => {
    fc.assert(
      fc.property(smallFloat, (a) => {
        const x = new Scalar(a);
        expect(x.neg().neg().data).toBe(a);
      })
    );
  });

  test("distributive law: z * (x + y) == z*x + z*y", () => {
    fc.assert(
      fc.property(smallFloat, smallFloat, smallFloat, (a, b, c) => {
        const x = new Scalar(a);
        const y = new Scalar(b);
        const z = new Scalar(c);
        
        const left = z.mul(x.add(y));
        const right = z.mul(x).add(z.mul(y));
        assertClose(left.data, right.data, 4);
      })
    );
  });

  test("subtraction as add + neg: x - y == x + (-y)", () => {
    fc.assert(
      fc.property(smallFloat, smallFloat, (a, b) => {
        const x = new Scalar(a);
        const y = new Scalar(b);
        assertClose(x.sub(y).data, x.add(y.neg()).data);
      })
    );
  });

  test("exp and log are inverses: log(exp(x)) ≈ x", () => {
    // Use bounded range to avoid overflow. EPS in log causes ~1e-6 error.
    const boundedFloat = fc.double({ noNaN: true, min: -5, max: 10 });
    fc.assert(
      fc.property(boundedFloat, (a) => {
        const x = new Scalar(a);
        // Looser tolerance (2 digits) due to EPS in operators.log
        assertClose(x.exp().log().data, a, 2);
      })
    );
  });

  test("sigmoid complement: 1 - sigmoid(x) ≈ sigmoid(-x)", () => {
    fc.assert(
      fc.property(smallFloat, (a) => {
        const x = new Scalar(a);
        const left = new Scalar(1).sub(x.sigmoid());
        const right = x.neg().sigmoid();
        assertClose(left.data, right.data, 4);
      })
    );
  });
});