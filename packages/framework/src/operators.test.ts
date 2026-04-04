// fc generates suites of inputs
import { test, fc } from '@fast-check/jest';
import { describe, expect } from '@jest/globals';
import {
    mul, 
    add, 
    neg, 
    max, 
    inv, 
    id,
    lt,
    eq,
    relu,
    reluBack,
    sigmoid,
    negList,
    addLists,
    sum,
    prod
} from "./operators.js";

const DIGIT_TOLERANCE = 5;

/** floats from set of small finite floats to avoid inputs that are false failures like NaN/Infinity, extreme over/underflow. */
const smallFloat = fc.double({ noNaN: true, min: -1000, max: 1000 });

/** A looser float comparator for nonlinear ops (sigmoid etc.). */
function assertClose(actual: number, expected: number, digits = DIGIT_TOLERANCE) {
  expect(actual).toBeCloseTo(expected, digits);
}

// task 0.1 - basic hypothesis tests
test('All operators return same value as JavaScript version', () => {
    fc.property(smallFloat, smallFloat, (x, y) => {
        expect(mul(x,y)).toBe(x * y);
        expect(add(x,y)).toBe(x + y);
        expect(neg(x)).toBe(-x);
        expect(max(x,y)).toBe(x > y ? x : y);

        if (Math.abs(x) > 1e-5) {
            expect(inv(x)).toBe(1.0 / x);
        }
    })
}) 

test("relu matches spec", () => {
  fc.assert(
    fc.property(smallFloat, (a) => {
      if (a > 0) expect(relu(a)).toBe(a);
      if (a < 0) expect(relu(a)).toBe(0.0);
      // if a === 0, either output is fine as long as your relu is consistent,
      // but typical is 0.0:
      if (a === 0) expect(relu(a)).toBe(0.0);
    })
  );
});


test("reluBack matches spec", () => {
  fc.assert(
    fc.property(smallFloat, smallFloat, (a, b) => {
      if (a > 0) expect(reluBack(a, b)).toBe(b);
      if (a < 0) expect(reluBack(a, b)).toBe(0.0);
      if (a === 0) {
        // often 0.0 at kink; if yours differs, change this line
        expect(reluBack(a, b)).toBe(0.0);
      }
    })
  );
});

test("id returns input", () => {
  fc.assert(fc.property(smallFloat, (a) => expect(id(a)).toBe(a)));
});


test("lt behaves like 'a < b' returning 1.0 else 0.0", () => {
  fc.assert(
    fc.property(smallFloat, (a) => {
      expect(lt(a - 1.0, a)).toBe(1.0);
      expect(lt(a, a - 1.0)).toBe(0.0);
    })
  );
});

test("max behavior on ordered inputs", () => {
  fc.assert(
    fc.property(smallFloat, (a) => {
      expect(max(a - 1.0, a)).toBe(a);
      expect(max(a, a - 1.0)).toBe(a);
      expect(max(a + 1.0, a)).toBe(a + 1.0);
      expect(max(a, a + 1.0)).toBe(a + 1.0);
    })
  );
});

test("eq behaves like equality returning 1.0 else 0.0", () => {
  fc.assert(
    fc.property(smallFloat, (a) => {
      expect(eq(a, a)).toBe(1.0);
      expect(eq(a, a - 1.0)).toBe(0.0);
      expect(eq(a, a + 1.0)).toBe(0.0);
    })
  );
});

// task 0.2 - property testing
test("sigmoid properties", () => {
  fc.assert(
    fc.property(smallFloat, (a) => {
      const s = sigmoid(a);

      // always between 0 and 1
      expect(s).toBeGreaterThanOrEqual(0.0);
      expect(s).toBeLessThanOrEqual(1.0);

      // 1 - sigmoid(a) == sigmoid(-a)
      assertClose(1.0 - sigmoid(a), sigmoid(-a), 6);

      // crosses 0 at 0.5 (sigmoid(0) == 0.5)
      assertClose(sigmoid(0.0), 0.5, 10);
    })
  );

  fc.assert(
    fc.property(smallFloat, smallFloat, (x, y) => {
      const a = Math.min(x, y);
      const b = Math.max(x, y);

      // Mathematically should be strictly increasing; the differece should be < 0 - but due to floating point the diff can be 0. So loosened requirements
      expect(sigmoid(a)).toBeLessThanOrEqual(sigmoid(b));
    })
  );
});

test("transitive property of lt: (a<b and b<c) => a<c", () => {
  fc.assert(
    fc.property(smallFloat, smallFloat, smallFloat, (a, b, c) => {
      const ab = lt(a, b) === 1.0;
      const bc = lt(b, c) === 1.0;

      if (ab && bc) {
        expect(lt(a, c)).toBe(1.0);
      }
    })
  );
});

test("mul is symmetric (commutative): mul(x,y) == mul(y,x)", () => {
  fc.assert(
    fc.property(smallFloat, smallFloat, (x, y) => {
      // exact should usually hold for same floating ops in JS,
      // but feel free to switch to assertClose if you want.
      expect(mul(x, y)).toBe(mul(y, x));
    })
  );
});

test("distributive law: z*(x+y) == z*x + z*y", () => {
  fc.assert(
    fc.property(smallFloat, smallFloat, smallFloat, (x, y, z) => {
      const left = mul(z, add(x, y));
      const right = add(mul(z, x), mul(z, y));
      assertClose(left, right, 6);
    })
  );
});

test("other property: double negation neg(neg(x)) == x", () => {
  fc.assert(
    fc.property(smallFloat, (x) => {
      expect(neg(neg(x))).toBe(x);
    })
  );
});

// task 0.3 - higher order functions


test("addLists elementwise matches JS", () => {
  fc.assert(
    fc.property(smallFloat, smallFloat, smallFloat, smallFloat, (a, b, c, d) => {
      const [x1, x2] = addLists([a, b], [c, d]);
      expect(x1).toBe(a + c);
      expect(x2).toBe(b + d);
    })
  );
});

test("sum distributes over elementwise add: sum(ls1)+sum(ls2) == sum(addLists(ls1,ls2))", () => {
  const len5 = fc.array(smallFloat, { minLength: 5, maxLength: 5 });

  fc.assert(
    fc.property(len5, len5, (ls1, ls2) => {
      const left = sum(ls1) + sum(ls2);
      const combined = addLists(ls1, ls2); // elementwise
      const right = sum(combined);
      assertClose(left, right, 6);
    })
  );
});

test("sum matches JS reduce", () => {
  fc.assert(
    fc.property(fc.array(smallFloat), (ls) => {
      const jsSum = ls.reduce((acc, v) => acc + v, 0.0);
      assertClose(sum(ls), jsSum, 6);
    })
  );
});

test("prod matches JS multiplication for length-3 list", () => {
  fc.assert(
    fc.property(smallFloat, smallFloat, smallFloat, (x, y, z) => {
      assertClose(prod([x, y, z]), x * y * z, 6);
    })
  );
});

test("negList negates each element", () => {
  fc.assert(
    fc.property(fc.array(smallFloat), (ls) => {
      const out = negList(ls);
      expect(out).toHaveLength(ls.length);
      for (let i = 0; i < ls.length; i++) {
        expect(out[i]).toBe(-ls[i]);
      }
    })
  );
});