import {
    Tensor,
} from '../dist/index.js';
import { assert, assertClose, section } from './helpers.js';

// ============================================================
// Tensor creation
// ============================================================

section('Tensor creation');

const z = Tensor.zeros([2, 3]);
assert(z.shape[0] === 2 && z.shape[1] === 3, 'zeros shape');
assert(z.toFloat32().every((v: number) => v === 0), 'zeros values');

const o = Tensor.ones([3]);
assert(o.toFloat32().every((v: number) => v === 1), 'ones values');
assert(o.shape[0] === 3, 'ones shape');

const r = Tensor.rand([100]);
const rd = r.toFloat32();
assert(rd.every((v: number) => v >= 0 && v <= 1), 'rand range [0,1]');
assert(r.shape[0] === 100, 'rand shape');

const rn = Tensor.randn([1000]);
assert(rn.shape[0] === 1000, 'randn shape');
const rnData = rn.toFloat32();
const rnMean = rnData.reduce((a: number, b: number) => a + b, 0) / rnData.length;
assert(Math.abs(rnMean) < 0.2, 'randn roughly zero-mean');

const f32 = Tensor.fromFloat32(new Float32Array([1.5, 2.5, 3.5]), [3]);
assert(f32.shape[0] === 3, 'fromFloat32 shape');
assertClose(f32.toFloat32()[0], 1.5, 1e-6, 'fromFloat32 value');

const multi = Tensor.fromFloat32(new Float32Array(24), [2, 3, 4]);
assert(multi.shape.length === 3, 'fromFloat32 3D dims');
assert(multi.shape[0] === 2 && multi.shape[1] === 3 && multi.shape[2] === 4, 'fromFloat32 3D shape');

// ============================================================
// Tensor properties: shape, size, dims
// ============================================================

section('Tensor properties');

const tp = Tensor.fromFloat32(new Float32Array(12), [3, 4]);
assert(tp.size === 12, 'size = product of shape');
assert(tp.dims === 2, 'dims = number of dimensions');
assert(tp.shape[0] === 3, 'shape[0]');
assert(tp.shape[1] === 4, 'shape[1]');

const scalar = Tensor.fromFloat32(new Float32Array([42]), [1]);
assert(scalar.size === 1, 'scalar size');

// ============================================================
// Basic elementwise ops
// ============================================================

section('Basic elementwise ops');

const a = Tensor.fromFloat32(new Float32Array([1, 2, 3]), [3]);
const b = Tensor.fromFloat32(new Float32Array([4, 5, 6]), [3]);

const addResult = a.add(b).toFloat32();
assert(addResult[0] === 5 && addResult[1] === 7 && addResult[2] === 9, 'add tensor+tensor');

const addScalar = a.add(10).toFloat32();
assert(addScalar[0] === 11 && addScalar[1] === 12 && addScalar[2] === 13, 'add tensor+scalar');

const subResult = a.sub(b).toFloat32();
assert(subResult[0] === -3 && subResult[1] === -3 && subResult[2] === -3, 'sub tensor-tensor');

const subScalar = b.sub(1).toFloat32();
assert(subScalar[0] === 3 && subScalar[1] === 4 && subScalar[2] === 5, 'sub tensor-scalar');

const mulResult = a.mul(b).toFloat32();
assert(mulResult[0] === 4 && mulResult[1] === 10 && mulResult[2] === 18, 'mul tensor*tensor');

const mulScalar = a.mul(3).toFloat32();
assert(mulScalar[0] === 3 && mulScalar[1] === 6 && mulScalar[2] === 9, 'mul tensor*scalar');

const negResult = a.neg().toFloat32();
assert(negResult[0] === -1 && negResult[1] === -2 && negResult[2] === -3, 'neg');

const divA = Tensor.fromFloat32(new Float32Array([6, 10, 15]), [3]);
const divB = Tensor.fromFloat32(new Float32Array([2, 5, 3]), [3]);
const divResult = divA.div(divB).toFloat32();
assert(divResult[0] === 3 && divResult[1] === 2 && divResult[2] === 5, 'div tensor/tensor');

const divScalar = divA.div(2).toFloat32();
assertClose(divScalar[0], 3, 1e-4, 'div tensor/scalar');

const powResult = a.pow(2).toFloat32();
assertClose(powResult[0], 1, 1e-4, 'pow 1^2');
assertClose(powResult[1], 4, 1e-4, 'pow 2^2');
assertClose(powResult[2], 9, 1e-4, 'pow 3^2');

const powHalf = Tensor.fromFloat32(new Float32Array([4, 9, 16]), [3]).pow(0.5).toFloat32();
assertClose(powHalf[0], 2, 1e-3, 'pow sqrt(4)');
assertClose(powHalf[1], 3, 1e-3, 'pow sqrt(9)');

// ============================================================
// Comparison ops
// ============================================================

section('Comparison ops');

const ltResult = a.lt(b).toFloat32();
assert(ltResult.every((v: number) => v === 1), 'lt: a < b all true');

const ltSelf = a.lt(a).toFloat32();
assert(ltSelf.every((v: number) => v === 0), 'lt: a < a all false');

const gtResult = b.gt(a).toFloat32();
assert(gtResult.every((v: number) => v === 1), 'gt: b > a all true');

const eqResult = a.eq(a).toFloat32();
assert(eqResult.every((v: number) => v === 1), 'eq: a == a all true');

const eqDiff = a.eq(b).toFloat32();
assert(eqDiff.every((v: number) => v === 0), 'eq: a == b all false');

const closeResult = a.isClose(a).toFloat32();
assert(closeResult.every((v: number) => v === 1), 'isClose: same tensor');

const almostSame = Tensor.fromFloat32(new Float32Array([1.000001, 2.000001, 3.000001]), [3]);
const isCloseSmall = a.isClose(almostSame, 1e-4).toFloat32();
assert(isCloseSmall.every((v: number) => v === 1), 'isClose: within tolerance');

// ============================================================
// Math ops: exp, log, sigmoid
// ============================================================

section('Math ops: exp, log, sigmoid');

const expInput = Tensor.fromFloat32(new Float32Array([0, 1, 2]), [3]);
const expResult = expInput.exp().toFloat32();
assertClose(expResult[0], 1.0, 1e-4, 'exp(0) = 1');
assertClose(expResult[1], Math.E, 1e-4, 'exp(1) = e');
assertClose(expResult[2], Math.E * Math.E, 1e-3, 'exp(2) = e^2');

const logInput = Tensor.fromFloat32(new Float32Array([1, Math.E, Math.E * Math.E]), [3]);
const logResult = logInput.log().toFloat32();
assertClose(logResult[0], 0.0, 1e-4, 'log(1) = 0');
assertClose(logResult[1], 1.0, 1e-4, 'log(e) = 1');
assertClose(logResult[2], 2.0, 1e-4, 'log(e^2) = 2');

const sigInput = Tensor.fromFloat32(new Float32Array([0, 100, -100]), [3]);
const sigResult = sigInput.sigmoid().toFloat32();
assertClose(sigResult[0], 0.5, 1e-4, 'sigmoid(0) = 0.5');
assertClose(sigResult[1], 1.0, 1e-2, 'sigmoid(100) ~ 1');
assertClose(sigResult[2], 0.0, 1e-2, 'sigmoid(-100) ~ 0');

// ============================================================
// Reductions
// ============================================================

section('Reductions');

const mat = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);

const sumDim1 = mat.sum(1).toFloat32();
assertClose(sumDim1[0], 6, 1e-4, 'sum dim=1 row0');
assertClose(sumDim1[1], 15, 1e-4, 'sum dim=1 row1');

const sumDim0 = mat.sum(0).toFloat32();
assertClose(sumDim0[0], 5, 1e-4, 'sum dim=0 col0');
assertClose(sumDim0[1], 7, 1e-4, 'sum dim=0 col1');
assertClose(sumDim0[2], 9, 1e-4, 'sum dim=0 col2');

const sumAll = mat.sum().toFloat32();
assertClose(sumAll[0], 21, 1e-4, 'sum all');

const meanDim1 = mat.mean(1).toFloat32();
assertClose(meanDim1[0], 2, 1e-4, 'mean dim=1 row0');
assertClose(meanDim1[1], 5, 1e-4, 'mean dim=1 row1');

const meanAll = mat.mean().toFloat32();
assertClose(meanAll[0], 3.5, 1e-4, 'mean all');

const maxDim1 = mat.max(1).toFloat32();
assertClose(maxDim1[0], 3, 1e-4, 'max dim=1 row0');
assertClose(maxDim1[1], 6, 1e-4, 'max dim=1 row1');

const maxDim0 = mat.max(0).toFloat32();
assertClose(maxDim0[0], 4, 1e-4, 'max dim=0 col0');
assertClose(maxDim0[1], 5, 1e-4, 'max dim=0 col1');
assertClose(maxDim0[2], 6, 1e-4, 'max dim=0 col2');

// ============================================================
// Layout: view, permute, contiguous
// ============================================================

section('Layout ops');

const layoutMat = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);

const viewed = layoutMat.view(3, 2);
assert(viewed.shape[0] === 3 && viewed.shape[1] === 2, 'view shape [2,3] -> [3,2]');
const viewedData = viewed.toFloat32();
assertClose(viewedData[0], 1, 1e-6, 'view preserves data order');
assertClose(viewedData[5], 6, 1e-6, 'view preserves last element');

const viewed1D = layoutMat.view(6);
assert(viewed1D.shape[0] === 6 && viewed1D.dims === 1, 'view flatten to 1D');

const perm = layoutMat.permute(1, 0);
assert(perm.shape[0] === 3 && perm.shape[1] === 2, 'permute transposes [2,3] -> [3,2]');

const t3d = Tensor.rand([2, 3, 4]);
const perm3d = t3d.permute(2, 0, 1);
assert(perm3d.shape[0] === 4 && perm3d.shape[1] === 2 && perm3d.shape[2] === 3, 'permute 3D');

const contig = perm.contiguous();
assert(contig.shape[0] === 3 && contig.shape[1] === 2, 'contiguous preserves shape');

// ============================================================
// MatMul
// ============================================================

section('MatMul');

const m1 = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4]), [2, 2]);
const m2 = Tensor.fromFloat32(new Float32Array([5, 6, 7, 8]), [2, 2]);
const mm = m1.matmul(m2).toFloat32();
assertClose(mm[0], 19, 1e-3, 'matmul [0,0] = 1*5+2*7');
assertClose(mm[1], 22, 1e-3, 'matmul [0,1] = 1*6+2*8');
assertClose(mm[2], 43, 1e-3, 'matmul [1,0] = 3*5+4*7');
assertClose(mm[3], 50, 1e-3, 'matmul [1,1] = 3*6+4*8');

const mA = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
const mB = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), [3, 2]);
const mmNonSq = mA.matmul(mB);
assert(mmNonSq.shape[0] === 2 && mmNonSq.shape[1] === 2, 'matmul [2,3]x[3,2] -> [2,2]');
const mmNonSqData = mmNonSq.toFloat32();
assertClose(mmNonSqData[0], 22, 1e-3, 'matmul non-square [0,0]');

const bmA = Tensor.rand([4, 3, 5]);
const bmB = Tensor.rand([4, 5, 2]);
const bmm = bmA.matmul(bmB);
assert(bmm.shape[0] === 4 && bmm.shape[1] === 3 && bmm.shape[2] === 2, 'batched matmul shape');

// ============================================================
// Broadcasting
// ============================================================

section('Broadcasting');

const bcastA = Tensor.fromFloat32(new Float32Array([1, 2, 3]), [3]);
const bcastScalar = Tensor.fromFloat32(new Float32Array([10]), [1]);
const bcastAdd = bcastA.add(bcastScalar).toFloat32();
assertClose(bcastAdd[0], 11, 1e-4, 'broadcast add [3]+[1] [0]');
assertClose(bcastAdd[2], 13, 1e-4, 'broadcast add [3]+[1] [2]');

const bcastMul = bcastA.mul(bcastScalar).toFloat32();
assertClose(bcastMul[0], 10, 1e-4, 'broadcast mul [3]*[1] [0]');
assertClose(bcastMul[2], 30, 1e-4, 'broadcast mul [3]*[1] [2]');

// ============================================================
// Utilities: clone, detach, toString, item, get, free
// ============================================================

section('Utilities');

const orig = Tensor.fromFloat32(new Float32Array([10, 20, 30]), [3]);

const cloned = orig.clone();
assert(cloned.shape[0] === 3, 'clone shape');
assertClose(cloned.toFloat32()[0], 10, 1e-6, 'clone values');
assertClose(cloned.toFloat32()[2], 30, 1e-6, 'clone last value');

const detached = orig.detach();
assert(detached.shape[0] === 3, 'detach shape');
assertClose(detached.toFloat32()[1], 20, 1e-6, 'detach values');

const str = orig.toString();
assert(str.includes('Tensor'), 'toString contains "Tensor"');
assert(str.includes('3'), 'toString contains shape');

const itemVal = Tensor.fromFloat32(new Float32Array([42.5]), [1]).item();
assertClose(itemVal, 42.5, 1e-4, 'item() returns scalar value');

const getMat = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
assertClose(getMat.get([0, 0]), 1, 1e-6, 'get([0,0])');
assertClose(getMat.get([0, 2]), 3, 1e-6, 'get([0,2])');
assertClose(getMat.get([1, 1]), 5, 1e-6, 'get([1,1])');

const toFree = Tensor.rand([10]);
toFree.free();

const dataStub = Tensor.fromFloat32(new Float32Array([1, 2]), [2]);
const storage = dataStub.data.storage;
assert(storage instanceof Float32Array, 'data.storage returns Float32Array');
assertClose(storage[0], 1, 1e-6, 'data.storage values');

