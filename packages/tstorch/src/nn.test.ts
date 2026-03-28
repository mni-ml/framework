import { describe, test, expect } from '@jest/globals';
import { tile, avgpool2d, maxpool2d } from './nn';
import { Tensor } from './tensor';

function assertClose(actual: number, expected: number, tolerance = 1e-5) {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);
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