import { describe, test, expect } from '@jest/globals';
import { avgpool2d, maxpool2d } from './nn';
import { Tensor } from './tensor';

// Helper function to assert close values
function assertClose(actual: number, expected: number, tolerance = 1e-5) {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);
}

describe('avgpool2d', () => {
  test('should compute average pooling correctly', () => {
    const t = Tensor.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);

    let out = avgpool2d(t, [2, 2]);
    assertClose(out.get([0, 0, 0, 0]), (1 + 2 + 5 + 6) / 4);

    out = avgpool2d(t, [2, 1]);
    assertClose(out.get([0, 0, 0, 0]), (1 + 5) / 2);

    out = avgpool2d(t, [1, 2]);
    assertClose(out.get([0, 0, 0, 0]), (1 + 2) / 2);
  });
});

describe('maxpool2d', () => {
  test('should compute max pooling correctly', () => {
    const t = Tensor.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);

    let out = maxpool2d(t, [2, 2]);
    assertClose(out.get([0, 0, 0, 0]), Math.max(1, 2, 5, 6));

    out = maxpool2d(t, [2, 1]);
    assertClose(out.get([0, 0, 0, 0]), Math.max(1, 5));

    out = maxpool2d(t, [1, 2]);
    assertClose(out.get([0, 0, 0, 0]), Math.max(1, 2));
  });
});

// describe('dropout', () => {
//   test('should apply dropout correctly', () => {
//     const t = Tensor.fromArray([1, 2, 3, 4]);

//     let q = dropout(t, 0.0);
//     expect(q.toArray()).toEqual(t.toArray());

//     q = dropout(t, 1.0);
//     expect(q.toArray().every((val) => val === 0)).toBe(true);

//     q = dropout(t, 1.0, true);
//     expect(q.toArray()).toEqual(t.toArray());
//   });
// });

// describe('softmax', () => {
//   test('should compute softmax correctly', () => {
//     const t = Tensor.fromArray([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);

//     let q = softmax(t, 3);
//     let x = q.sum(3);
//     assertClose(x.get(0, 0, 0, 0), 1.0);

//     q = softmax(t, 1);
//     x = q.sum(1);
//     assertClose(x.get(0, 0, 0, 0), 1.0);
//   });
// });

// describe('logsoftmax', () => {
//   test('should compute logsoftmax correctly', () => {
//     const t = Tensor.fromArray([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);

//     const q = softmax(t, 3);
//     const q2 = logsoftmax(t, 3).exp();

//     q.indices().forEach((idx) => {
//       assertClose(q.get(...idx), q2.get(...idx));
//     });
//   });
// });