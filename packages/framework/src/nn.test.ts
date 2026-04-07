import { describe, test, expect } from '@jest/globals';
import { tile, avgpool2d, maxpool2d, softmax, logsoftmax, dropout } from './nn.js';
import { Tensor } from './tensor.js';

function assertClose(actual: number, expected: number, tolerance = 1e-5) {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);
}

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

// ============================================================
// tile()
// ============================================================

describe('tile', () => {
  test('output shape matches spec: [B, C, nH, nW, kh*kw]', () => {
    const t = Tensor.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);
    const [tiled, nH, nW] = tile(t, [2, 2]);
    expect(tiled.shape).toEqual([1, 1, 2, 2, 4]);
    expect(nH).toBe(2);
    expect(nW).toBe(2);
  });

  test('non-square kernel', () => {
    const t = Tensor.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);

    const [tiled2x1, nH2, nW2] = tile(t, [2, 1]);
    expect(tiled2x1.shape).toEqual([1, 1, 2, 4, 2]);
    expect(nH2).toBe(2);
    expect(nW2).toBe(4);

    const [tiled1x2, nH1, nW1] = tile(t, [1, 2]);
    expect(tiled1x2.shape).toEqual([1, 1, 4, 2, 2]);
    expect(nH1).toBe(4);
    expect(nW1).toBe(2);
  });

  test('kernel = full input size (global pooling tile)', () => {
    const t = Tensor.tensor([[[[1, 2], [3, 4]]]]);
    const [tiled, nH, nW] = tile(t, [2, 2]);
    expect(tiled.shape).toEqual([1, 1, 1, 1, 4]);
    expect(nH).toBe(1);
    expect(nW).toBe(1);
  });

  test('tiled values contain the correct pooling windows', () => {
    // 1  2 | 3  4
    // 5  6 | 7  8
    // -----+-----
    // 9 10 | 11 12
    // 13 14| 15 16
    const t = Tensor.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);
    const [tiled] = tile(t, [2, 2]);

    // Top-left 2x2 window: [1, 2, 5, 6]
    expect(tiled.get([0, 0, 0, 0, 0])).toBe(1);
    expect(tiled.get([0, 0, 0, 0, 1])).toBe(2);
    expect(tiled.get([0, 0, 0, 0, 2])).toBe(5);
    expect(tiled.get([0, 0, 0, 0, 3])).toBe(6);

    // Top-right 2x2 window: [3, 4, 7, 8]
    expect(tiled.get([0, 0, 0, 1, 0])).toBe(3);
    expect(tiled.get([0, 0, 0, 1, 1])).toBe(4);
    expect(tiled.get([0, 0, 0, 1, 2])).toBe(7);
    expect(tiled.get([0, 0, 0, 1, 3])).toBe(8);

    // Bottom-left 2x2 window: [9, 10, 13, 14]
    expect(tiled.get([0, 0, 1, 0, 0])).toBe(9);
    expect(tiled.get([0, 0, 1, 0, 1])).toBe(10);
    expect(tiled.get([0, 0, 1, 0, 2])).toBe(13);
    expect(tiled.get([0, 0, 1, 0, 3])).toBe(14);

    // Bottom-right 2x2 window: [11, 12, 15, 16]
    expect(tiled.get([0, 0, 1, 1, 0])).toBe(11);
    expect(tiled.get([0, 0, 1, 1, 1])).toBe(12);
    expect(tiled.get([0, 0, 1, 1, 2])).toBe(15);
    expect(tiled.get([0, 0, 1, 1, 3])).toBe(16);
  });

  test('throws on indivisible dimensions', () => {
    const t = Tensor.tensor([[[[1, 2, 3], [4, 5, 6]]]]);
    expect(() => tile(t, [2, 2])).toThrow(/divisible/);
  });
});

// ============================================================
// avgpool2d
// ============================================================

describe('avgpool2d', () => {
  test('2x2 kernel on 4x4 input — all output values', () => {
    const t = Tensor.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);
    const out = avgpool2d(t, [2, 2]);

    expect(out.shape).toEqual([1, 1, 2, 2]);
    assertClose(out.get([0, 0, 0, 0]), (1 + 2 + 5 + 6) / 4);
    assertClose(out.get([0, 0, 0, 1]), (3 + 4 + 7 + 8) / 4);
    assertClose(out.get([0, 0, 1, 0]), (9 + 10 + 13 + 14) / 4);
    assertClose(out.get([0, 0, 1, 1]), (11 + 12 + 15 + 16) / 4);
  });

  test('2x1 kernel (pool only over height)', () => {
    const t = Tensor.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);
    const out = avgpool2d(t, [2, 1]);

    expect(out.shape).toEqual([1, 1, 2, 4]);
    assertClose(out.get([0, 0, 0, 0]), (1 + 5) / 2);
    assertClose(out.get([0, 0, 0, 1]), (2 + 6) / 2);
  });

  test('1x2 kernel (pool only over width)', () => {
    const t = Tensor.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);
    const out = avgpool2d(t, [1, 2]);

    expect(out.shape).toEqual([1, 1, 4, 2]);
    assertClose(out.get([0, 0, 0, 0]), (1 + 2) / 2);
    assertClose(out.get([0, 0, 0, 1]), (3 + 4) / 2);
  });

  test('global average pooling (kernel = input size)', () => {
    const t = Tensor.tensor([[[[1, 2], [3, 4]]]]);
    const out = avgpool2d(t, [2, 2]);
    expect(out.shape).toEqual([1, 1, 1, 1]);
    assertClose(out.get([0, 0, 0, 0]), (1 + 2 + 3 + 4) / 4);
  });

  test('multi-batch', () => {
    const t = Tensor.tensor([
      [[[1, 2], [3, 4]]],
      [[[10, 20], [30, 40]]],
    ]);
    const out = avgpool2d(t, [2, 2]);
    expect(out.shape).toEqual([2, 1, 1, 1]);
    assertClose(out.get([0, 0, 0, 0]), (1 + 2 + 3 + 4) / 4);
    assertClose(out.get([1, 0, 0, 0]), (10 + 20 + 30 + 40) / 4);
  });

  test('multi-channel', () => {
    const t = Tensor.tensor([[
      [[1, 2], [3, 4]],
      [[10, 20], [30, 40]],
    ]]);
    const out = avgpool2d(t, [2, 2]);
    expect(out.shape).toEqual([1, 2, 1, 1]);
    assertClose(out.get([0, 0, 0, 0]), (1 + 2 + 3 + 4) / 4);
    assertClose(out.get([0, 1, 0, 0]), (10 + 20 + 30 + 40) / 4);
  });
});

// ============================================================
// maxpool2d
// ============================================================

describe('maxpool2d', () => {
  test('2x2 kernel on 4x4 input — all output values', () => {
    const t = Tensor.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);
    const out = maxpool2d(t, [2, 2]);

    expect(out.shape).toEqual([1, 1, 2, 2]);
    assertClose(out.get([0, 0, 0, 0]), Math.max(1, 2, 5, 6));
    assertClose(out.get([0, 0, 0, 1]), Math.max(3, 4, 7, 8));
    assertClose(out.get([0, 0, 1, 0]), Math.max(9, 10, 13, 14));
    assertClose(out.get([0, 0, 1, 1]), Math.max(11, 12, 15, 16));
  });

  test('2x1 kernel', () => {
    const t = Tensor.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);
    const out = maxpool2d(t, [2, 1]);

    expect(out.shape).toEqual([1, 1, 2, 4]);
    assertClose(out.get([0, 0, 0, 0]), Math.max(1, 5));
    assertClose(out.get([0, 0, 0, 1]), Math.max(2, 6));
  });

  test('1x2 kernel', () => {
    const t = Tensor.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);
    const out = maxpool2d(t, [1, 2]);

    expect(out.shape).toEqual([1, 1, 4, 2]);
    assertClose(out.get([0, 0, 0, 0]), Math.max(1, 2));
    assertClose(out.get([0, 0, 0, 1]), Math.max(3, 4));
  });

  test('global max pooling', () => {
    const t = Tensor.tensor([[[[1, 2], [3, 4]]]]);
    const out = maxpool2d(t, [2, 2]);
    expect(out.shape).toEqual([1, 1, 1, 1]);
    assertClose(out.get([0, 0, 0, 0]), 4);
  });

  test('multi-batch', () => {
    const t = Tensor.tensor([
      [[[1, 2], [3, 4]]],
      [[[40, 30], [20, 10]]],
    ]);
    const out = maxpool2d(t, [2, 2]);
    expect(out.shape).toEqual([2, 1, 1, 1]);
    assertClose(out.get([0, 0, 0, 0]), 4);
    assertClose(out.get([1, 0, 0, 0]), 40);
  });

  test('multi-channel', () => {
    const t = Tensor.tensor([[
      [[1, 2], [3, 4]],
      [[40, 30], [20, 10]],
    ]]);
    const out = maxpool2d(t, [2, 2]);
    expect(out.shape).toEqual([1, 2, 1, 1]);
    assertClose(out.get([0, 0, 0, 0]), 4);
    assertClose(out.get([0, 1, 0, 0]), 40);
  });

  test('works with negative values', () => {
    const t = Tensor.tensor([[[[-5, -3], [-1, -8]]]]);
    const out = maxpool2d(t, [2, 2]);
    expect(out.shape).toEqual([1, 1, 1, 1]);
    assertClose(out.get([0, 0, 0, 0]), -1);
  });
});

// ============================================================
// softmax
// ============================================================

describe('softmax', () => {
  test('output sums to 1 along dim', () => {
    const t = Tensor.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);

    const q = softmax(t, 3);
    const s = q.sum(3);
    for (let i = 0; i < 4; i++) {
      assertClose(s.get([0, 0, i, 0]), 1.0);
    }
  });

  test('output sums to 1 along dim=1', () => {
    const t = Tensor.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);

    const q = softmax(t, 1);
    const s = q.sum(1);
    assertClose(s.get([0, 0, 0, 0]), 1.0);
  });

  test('all values are non-negative', () => {
    const t = Tensor.tensor([[-10, -5, 0, 5, 10]]);
    const q = softmax(t, 1);
    for (let i = 0; i < 5; i++) {
      expect(q.get([0, i])).toBeGreaterThanOrEqual(0);
    }
  });

  test('larger inputs get larger probabilities', () => {
    const t = Tensor.tensor([[1, 2, 3]]);
    const q = softmax(t, 1);
    expect(q.get([0, 2])).toBeGreaterThan(q.get([0, 1]));
    expect(q.get([0, 1])).toBeGreaterThan(q.get([0, 0]));
  });

  test('numerically stable with large values', () => {
    const t = Tensor.tensor([[1000, 1001, 1002]]);
    const q = softmax(t, 1);
    const s = q.sum(1);
    assertClose(s.get([0, 0]), 1.0);
  });
});

// ============================================================
// logsoftmax
// ============================================================

describe('logsoftmax', () => {
  test('exp(logsoftmax) equals softmax', () => {
    const t = Tensor.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);

    const sm = softmax(t, 3);
    const lsm = logsoftmax(t, 3).exp();

    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        assertClose(sm.get([0, 0, i, j]), lsm.get([0, 0, i, j]));
      }
    }
  });

  test('all values are <= 0', () => {
    const t = Tensor.tensor([[1, 2, 3]]);
    const q = logsoftmax(t, 1);
    for (let i = 0; i < 3; i++) {
      expect(q.get([0, i])).toBeLessThanOrEqual(1e-7);
    }
  });

  test('numerically stable with large values', () => {
    const t = Tensor.tensor([[1000, 1001, 1002]]);
    const sm = softmax(t, 1);
    const lsm = logsoftmax(t, 1).exp();
    for (let i = 0; i < 3; i++) {
      assertClose(sm.get([0, i]), lsm.get([0, i]));
    }
  });
});

// ============================================================
// dropout
// ============================================================

describe('dropout', () => {
  test('rate=0 preserves all values', () => {
    const t = Tensor.tensor([[1, 2, 3, 4]]);
    const out = dropout(t, 0.0);
    for (let i = 0; i < 4; i++) {
      assertClose(out.get([0, i]), t.get([0, i]));
    }
  });

  test('rate=1 zeros everything', () => {
    const t = Tensor.tensor([[1, 2, 3, 4]]);
    const out = dropout(t, 1.0);
    for (let i = 0; i < 4; i++) {
      assertClose(out.get([0, i]), 0);
    }
  });

  test('ignore=true returns input unchanged regardless of rate', () => {
    const t = Tensor.tensor([[1, 2, 3, 4]]);
    const out = dropout(t, 1.0, true);
    for (let i = 0; i < 4; i++) {
      assertClose(out.get([0, i]), t.get([0, i]));
    }
  });

  test('output values are either 0 or scaled by 1/(1-rate)', () => {
    const t = Tensor.tensor([[5, 5, 5, 5, 5, 5, 5, 5, 5, 5]]);
    const rate = 0.5;
    const out = dropout(t, rate);
    const scale = 1 / (1 - rate);
    for (let i = 0; i < 10; i++) {
      const v = out.get([0, i]);
      expect(v === 0 || Math.abs(v - 5 * scale) < 1e-5).toBe(true);
    }
  });

  test('preserves shape', () => {
    const t = Tensor.rand([2, 3, 4]);
    const out = dropout(t, 0.3);
    expect(out.shape).toEqual([2, 3, 4]);
  });
});

// ============================================================
// backward pass (gradient checks)
// ============================================================

describe('avgpool2d backward', () => {
  test('gradient check — 2x2 kernel', () => {
    const input = Tensor.rand([1, 1, 4, 4]);
    tensorGradCheck((x) => avgpool2d(x, [2, 2]), input);
  });

  test('gradient check — multi-batch multi-channel', () => {
    const input = Tensor.rand([2, 2, 4, 4]);
    tensorGradCheck((x) => avgpool2d(x, [2, 2]), input);
  });

  test('gradient check — non-square kernel', () => {
    const input = Tensor.rand([1, 1, 4, 6]);
    tensorGradCheck((x) => avgpool2d(x, [2, 3]), input);
  });

  test('gradient check — global pooling', () => {
    const input = Tensor.rand([1, 1, 2, 2]);
    tensorGradCheck((x) => avgpool2d(x, [2, 2]), input);
  });
});

describe('maxpool2d backward', () => {
  test('gradient check — 2x2 kernel', () => {
    // Use distinct values to avoid ties at the max boundary
    const input = Tensor.tensor([[[[1, 5, 2, 8], [3, 7, 4, 6], [9, 2, 11, 3], [10, 1, 12, 4]]]]);
    tensorGradCheck((x) => maxpool2d(x, [2, 2]), input);
  });

  test('gradient check — multi-batch', () => {
    const input = Tensor.tensor([
      [[[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]]],
      [[[16, 12, 8, 4], [15, 11, 7, 3], [14, 10, 6, 2], [13, 9, 5, 1]]],
    ]);
    tensorGradCheck((x) => maxpool2d(x, [2, 2]), input);
  });

  test('gradient check — 2x1 kernel', () => {
    const input = Tensor.tensor([[[[1, 5, 9, 13], [3, 7, 11, 15], [2, 6, 10, 14], [4, 8, 12, 16]]]]);
    tensorGradCheck((x) => maxpool2d(x, [2, 1]), input);
  });
});

describe('softmax backward', () => {
  test('gradient check — dim=1 on 2D', () => {
    const input = Tensor.rand([1, 4]);
    tensorGradCheck((x) => softmax(x, 1), input);
  });

  test('gradient check — dim=3 on 4D', () => {
    const input = Tensor.rand([1, 1, 2, 3]);
    tensorGradCheck((x) => softmax(x, 3), input);
  });

  test('gradient check — dim=0', () => {
    const input = Tensor.rand([3, 2]);
    tensorGradCheck((x) => softmax(x, 0), input);
  });
});

describe('logsoftmax backward', () => {
  test('gradient check — dim=1 on 2D', () => {
    const input = Tensor.rand([1, 4]);
    tensorGradCheck((x) => logsoftmax(x, 1), input);
  });

  test('gradient check — dim=3 on 4D', () => {
    const input = Tensor.rand([1, 1, 2, 3]);
    tensorGradCheck((x) => logsoftmax(x, 3), input);
  });

  test('gradient check — dim=0', () => {
    const input = Tensor.rand([3, 2]);
    tensorGradCheck((x) => logsoftmax(x, 0), input);
  });
});

describe('dropout backward', () => {
  test('gradient check — rate=0 (identity)', () => {
    const input = Tensor.rand([2, 3]);
    tensorGradCheck((x) => dropout(x, 0.0), input);
  });

  test('gradient check — ignore=true (identity)', () => {
    const input = Tensor.rand([2, 3]);
    tensorGradCheck((x) => dropout(x, 0.5, true), input);
  });
});