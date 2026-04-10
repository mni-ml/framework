import {
    Tensor,
} from '../dist/index.js';
import { assert, assertClose, section } from './helpers.js';

// ============================================================
// Autograd / backward
// ============================================================

section('Autograd / backward');

// simple gradient: d/dx (x^2) at x=3 should be 6
const xGrad = Tensor.fromFloat32(new Float32Array([3]), [1]).setRequiresGrad(true);
const xSq = xGrad.pow(2);
xSq.backward();
const gradX = xGrad.grad;
assert(gradX !== null, 'gradient exists after backward');
assertClose(gradX!.toFloat32()[0], 6, 1e-3, 'd/dx(x^2) at x=3 = 6');

// gradient through mul
const paramA = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4]), [2, 2]).setRequiresGrad(true);
const paramB = Tensor.fromFloat32(new Float32Array([5, 6, 7, 8]), [2, 2]).setRequiresGrad(true);
const c = paramA.mul(paramB).sum(0).sum(0);
c.backward();
assert(paramA.grad !== null, 'paramA gradient exists');
assert(paramB.grad !== null, 'paramB gradient exists');

const gradAData = paramA.grad!.toFloat32();
assertClose(gradAData[0], 5, 1e-3, 'grad_a[0] = b[0]');
assertClose(gradAData[1], 6, 1e-3, 'grad_a[1] = b[1]');

// gradient through add
const addX = Tensor.fromFloat32(new Float32Array([2, 3]), [2]).setRequiresGrad(true);
const addY = Tensor.fromFloat32(new Float32Array([4, 5]), [2]).setRequiresGrad(true);
const addSum = addX.add(addY).sum();
addSum.backward();
assert(addX.grad !== null && addY.grad !== null, 'add gradients exist');
assertClose(addX.grad!.toFloat32()[0], 1, 1e-3, 'd/dx(x+y).sum() = 1');
assertClose(addY.grad!.toFloat32()[0], 1, 1e-3, 'd/dy(x+y).sum() = 1');

// gradient through matmul
const mmX = Tensor.fromFloat32(new Float32Array([1, 0, 0, 1]), [2, 2]).setRequiresGrad(true);
const mmY = Tensor.fromFloat32(new Float32Array([3, 4, 5, 6]), [2, 2]).setRequiresGrad(true);
const mmOut = mmX.matmul(mmY).sum();
mmOut.backward();
assert(mmX.grad !== null, 'matmul grad exists');

