import {
    Tensor, native,
    Module, Parameter,
    Linear, ReLU, Sigmoid, Tanh, Embedding,
    Conv1d, Conv2d,
    softmax, logsoftmax, gelu, dropout,
    crossEntropyLoss, mseLoss, layerNorm,
    flashAttention, residualLayerNorm, biasGelu,
    randRange, tile, avgpool2d, maxpool2d,
    SGD, Adam, GradScaler,
} from '../dist/index.js';

let passed = 0;
let failed = 0;
const failures = [];

function assert(cond, msg) {
    if (!cond) {
        failed++;
        failures.push(msg);
        console.error(`  FAIL: ${msg}`);
    } else {
        passed++;
    }
}

function assertClose(a, b, tol = 1e-4, msg = '') {
    if (Math.abs(a - b) > tol) {
        failed++;
        const detail = `${msg}: ${a} != ${b} (tol=${tol})`;
        failures.push(detail);
        console.error(`  FAIL: ${detail}`);
    } else {
        passed++;
    }
}

function section(name) {
    console.log(`\n--- ${name} ---`);
}

// ============================================================
// Tensor creation
// ============================================================

section('Tensor creation');

const z = Tensor.zeros([2, 3]);
assert(z.shape[0] === 2 && z.shape[1] === 3, 'zeros shape');
assert(z.toFloat32().every(v => v === 0), 'zeros values');

const o = Tensor.ones([3]);
assert(o.toFloat32().every(v => v === 1), 'ones values');
assert(o.shape[0] === 3, 'ones shape');

const r = Tensor.rand([100]);
const rd = r.toFloat32();
assert(rd.every(v => v >= 0 && v <= 1), 'rand range [0,1]');
assert(r.shape[0] === 100, 'rand shape');

const rn = Tensor.randn([1000]);
assert(rn.shape[0] === 1000, 'randn shape');
const rnData = rn.toFloat32();
const rnMean = rnData.reduce((a, b) => a + b, 0) / rnData.length;
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

// add
const addResult = a.add(b).toFloat32();
assert(addResult[0] === 5 && addResult[1] === 7 && addResult[2] === 9, 'add tensor+tensor');

// add scalar
const addScalar = a.add(10).toFloat32();
assert(addScalar[0] === 11 && addScalar[1] === 12 && addScalar[2] === 13, 'add tensor+scalar');

// sub
const subResult = a.sub(b).toFloat32();
assert(subResult[0] === -3 && subResult[1] === -3 && subResult[2] === -3, 'sub tensor-tensor');

// sub scalar
const subScalar = b.sub(1).toFloat32();
assert(subScalar[0] === 3 && subScalar[1] === 4 && subScalar[2] === 5, 'sub tensor-scalar');

// mul
const mulResult = a.mul(b).toFloat32();
assert(mulResult[0] === 4 && mulResult[1] === 10 && mulResult[2] === 18, 'mul tensor*tensor');

// mul scalar
const mulScalar = a.mul(3).toFloat32();
assert(mulScalar[0] === 3 && mulScalar[1] === 6 && mulScalar[2] === 9, 'mul tensor*scalar');

// neg
const negResult = a.neg().toFloat32();
assert(negResult[0] === -1 && negResult[1] === -2 && negResult[2] === -3, 'neg');

// div tensor
const divA = Tensor.fromFloat32(new Float32Array([6, 10, 15]), [3]);
const divB = Tensor.fromFloat32(new Float32Array([2, 5, 3]), [3]);
const divResult = divA.div(divB).toFloat32();
assert(divResult[0] === 3 && divResult[1] === 2 && divResult[2] === 5, 'div tensor/tensor');

// div scalar
const divScalar = divA.div(2).toFloat32();
assertClose(divScalar[0], 3, 1e-4, 'div tensor/scalar');

// pow
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
assert(ltResult.every(v => v === 1), 'lt: a < b all true');

const ltSelf = a.lt(a).toFloat32();
assert(ltSelf.every(v => v === 0), 'lt: a < a all false');

const gtResult = b.gt(a).toFloat32();
assert(gtResult.every(v => v === 1), 'gt: b > a all true');

const eqResult = a.eq(a).toFloat32();
assert(eqResult.every(v => v === 1), 'eq: a == a all true');

const eqDiff = a.eq(b).toFloat32();
assert(eqDiff.every(v => v === 0), 'eq: a == b all false');

const closeResult = a.isClose(a).toFloat32();
assert(closeResult.every(v => v === 1), 'isClose: same tensor');

const almostSame = Tensor.fromFloat32(new Float32Array([1.000001, 2.000001, 3.000001]), [3]);
const isCloseSmall = a.isClose(almostSame, 1e-4).toFloat32();
assert(isCloseSmall.every(v => v === 1), 'isClose: within tolerance');

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
// Activations: relu, gelu
// ============================================================

section('Activations');

const actInput = Tensor.fromFloat32(new Float32Array([-2, -1, 0, 1, 2]), [5]);

// relu
const reluOut = actInput.relu().toFloat32();
assert(reluOut[0] === 0 && reluOut[1] === 0, 'relu: negatives become 0');
assert(reluOut[2] === 0, 'relu: zero stays 0');
assert(reluOut[3] === 1 && reluOut[4] === 2, 'relu: positives unchanged');

// gelu
const geluOut = gelu(actInput).toFloat32();
assert(geluOut[0] <= 0, 'gelu(-2) <= 0');
assertClose(geluOut[2], 0, 1e-3, 'gelu(0) ~ 0');
assert(geluOut[4] > 1.5, 'gelu(2) > 1.5');

// ReLU module
const reluMod = new ReLU();
const reluModOut = reluMod.forward(actInput).toFloat32();
assert(reluModOut[0] === 0 && reluModOut[4] === 2, 'ReLU module');

// Sigmoid module
const sigMod = new Sigmoid();
const sigModOut = sigMod.forward(Tensor.fromFloat32(new Float32Array([0]), [1])).toFloat32();
assertClose(sigModOut[0], 0.5, 1e-4, 'Sigmoid module(0) = 0.5');

// Tanh module
const tanhMod = new Tanh();
const tanhOut = tanhMod.forward(Tensor.fromFloat32(new Float32Array([0]), [1])).toFloat32();
assertClose(tanhOut[0], 0, 1e-3, 'Tanh(0) ~ 0');

// ============================================================
// Reductions
// ============================================================

section('Reductions');

const mat = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);

// sum with dim
const sumDim1 = mat.sum(1).toFloat32();
assertClose(sumDim1[0], 6, 1e-4, 'sum dim=1 row0');
assertClose(sumDim1[1], 15, 1e-4, 'sum dim=1 row1');

const sumDim0 = mat.sum(0).toFloat32();
assertClose(sumDim0[0], 5, 1e-4, 'sum dim=0 col0');
assertClose(sumDim0[1], 7, 1e-4, 'sum dim=0 col1');
assertClose(sumDim0[2], 9, 1e-4, 'sum dim=0 col2');

// sum all
const sumAll = mat.sum().toFloat32();
assertClose(sumAll[0], 21, 1e-4, 'sum all');

// mean with dim
const meanDim1 = mat.mean(1).toFloat32();
assertClose(meanDim1[0], 2, 1e-4, 'mean dim=1 row0');
assertClose(meanDim1[1], 5, 1e-4, 'mean dim=1 row1');

// mean all
const meanAll = mat.mean().toFloat32();
assertClose(meanAll[0], 3.5, 1e-4, 'mean all');

// max with dim
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

// view
const viewed = layoutMat.view(3, 2);
assert(viewed.shape[0] === 3 && viewed.shape[1] === 2, 'view shape [2,3] -> [3,2]');
const viewedData = viewed.toFloat32();
assertClose(viewedData[0], 1, 1e-6, 'view preserves data order');
assertClose(viewedData[5], 6, 1e-6, 'view preserves last element');

const viewed1D = layoutMat.view(6);
assert(viewed1D.shape[0] === 6 && viewed1D.dims === 1, 'view flatten to 1D');

// permute
const perm = layoutMat.permute(1, 0);
assert(perm.shape[0] === 3 && perm.shape[1] === 2, 'permute transposes [2,3] -> [3,2]');

// permute 3D
const t3d = Tensor.rand([2, 3, 4]);
const perm3d = t3d.permute(2, 0, 1);
assert(perm3d.shape[0] === 4 && perm3d.shape[1] === 2 && perm3d.shape[2] === 3, 'permute 3D');

// contiguous
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

// non-square matmul
const mA = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
const mB = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), [3, 2]);
const mmNonSq = mA.matmul(mB);
assert(mmNonSq.shape[0] === 2 && mmNonSq.shape[1] === 2, 'matmul [2,3]x[3,2] -> [2,2]');
const mmNonSqData = mmNonSq.toFloat32();
assertClose(mmNonSqData[0], 22, 1e-3, 'matmul non-square [0,0]');

// batched matmul
const bmA = Tensor.rand([4, 3, 5]);
const bmB = Tensor.rand([4, 5, 2]);
const bmm = bmA.matmul(bmB);
assert(bmm.shape[0] === 4 && bmm.shape[1] === 3 && bmm.shape[2] === 2, 'batched matmul shape');

// ============================================================
// Softmax / LogSoftmax
// ============================================================

section('Softmax / LogSoftmax');

const smInput = Tensor.fromFloat32(new Float32Array([1, 2, 3, 1, 2, 3]), [2, 3]);
const smOut = softmax(smInput, 1);
const smData = smOut.toFloat32();

// each row sums to 1
assertClose(smData[0] + smData[1] + smData[2], 1.0, 1e-4, 'softmax row0 sums to 1');
assertClose(smData[3] + smData[4] + smData[5], 1.0, 1e-4, 'softmax row1 sums to 1');

// values are positive
assert(smData.every(v => v > 0), 'softmax all positive');

// larger input -> larger probability
assert(smData[2] > smData[1] && smData[1] > smData[0], 'softmax preserves ordering');

// logsoftmax
const lsmOut = logsoftmax(smInput, 1).toFloat32();
assert(lsmOut.every(v => v <= 0), 'logsoftmax all <= 0');
const expLsm = Math.exp(lsmOut[0]) + Math.exp(lsmOut[1]) + Math.exp(lsmOut[2]);
assertClose(expLsm, 1.0, 1e-3, 'exp(logsoftmax) sums to 1');

// ============================================================
// Loss functions
// ============================================================

section('Loss functions');

// cross-entropy loss
const ceLogits = Tensor.fromFloat32(new Float32Array([2, 1, 0.1, 0.1, 1, 2]), [2, 3]);
const ceTargets = [[0], [2]];
const ceLoss = crossEntropyLoss(ceLogits, ceTargets);
const ceLossVal = ceLoss.toFloat32()[0];
assert(ceLossVal > 0, 'crossEntropyLoss positive');
assert(ceLossVal < 5, 'crossEntropyLoss reasonable range');

// cross-entropy with perfect prediction should be low
const perfectLogits = Tensor.fromFloat32(new Float32Array([100, -100, -100, -100, -100, 100]), [2, 3]);
const perfectLoss = crossEntropyLoss(perfectLogits, [[0], [2]]).toFloat32()[0];
assert(perfectLoss < ceLossVal, 'perfect prediction has lower loss');

// MSE loss
const predMse = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4]), [2, 2]);
const targetMse = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4]), [2, 2]);
const mse = mseLoss(predMse, targetMse).toFloat32()[0];
assertClose(mse, 0, 1e-4, 'mseLoss identical = 0');

const predMse2 = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4]), [2, 2]);
const targetMse2 = Tensor.fromFloat32(new Float32Array([2, 3, 4, 5]), [2, 2]);
const mse2 = mseLoss(predMse2, targetMse2).toFloat32()[0];
assertClose(mse2, 1.0, 1e-4, 'mseLoss off-by-1 = 1.0');

// ============================================================
// LayerNorm
// ============================================================

section('LayerNorm');

const lnInput = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
const gamma = Tensor.ones([3]).setRequiresGrad(true);
const beta = Tensor.zeros([3]).setRequiresGrad(true);
const lnOut = layerNorm(lnInput, gamma, beta);
assert(lnOut.shape[0] === 2 && lnOut.shape[1] === 3, 'layerNorm shape preserved');

const lnData = lnOut.toFloat32();
// each row should be roughly zero-mean
const row0Mean = (lnData[0] + lnData[1] + lnData[2]) / 3;
assertClose(row0Mean, 0, 1e-3, 'layerNorm row0 zero mean');

const row1Mean = (lnData[3] + lnData[4] + lnData[5]) / 3;
assertClose(row1Mean, 0, 1e-3, 'layerNorm row1 zero mean');

// ============================================================
// Conv1d module
// ============================================================

section('Conv1d');

const conv1d = new Conv1d(3, 8, 3, 1, 1);
const conv1dInput = Tensor.rand([2, 3, 10]);
const conv1dOut = conv1d.forward(conv1dInput);
assert(conv1dOut.shape[0] === 2, 'conv1d batch preserved');
assert(conv1dOut.shape[1] === 8, 'conv1d out_channels');
assert(conv1dOut.shape[2] === 10, 'conv1d length preserved with padding=1');

// different stride
const conv1dS2 = new Conv1d(3, 4, 3, 2, 0);
const conv1dS2Out = conv1dS2.forward(Tensor.rand([1, 3, 10]));
assert(conv1dS2Out.shape[0] === 1 && conv1dS2Out.shape[1] === 4, 'conv1d stride=2 batch/channels');
assert(conv1dS2Out.shape[2] === 4, 'conv1d stride=2 length = (10-3)/2+1 = 4');

// conv1d has parameters
const conv1dParams = conv1d.parameters();
assert(conv1dParams.length === 2, 'conv1d has 2 parameters (weight + bias)');

// ============================================================
// Conv2d module
// ============================================================

section('Conv2d');

const conv2d = new Conv2d(3, 16, 3, 1, 1);
const conv2dInput = Tensor.rand([1, 3, 8, 8]);
const conv2dOut = conv2d.forward(conv2dInput);
assert(conv2dOut.shape[0] === 1, 'conv2d batch preserved');
assert(conv2dOut.shape[1] === 16, 'conv2d out_channels');
assert(conv2dOut.shape[2] === 8 && conv2dOut.shape[3] === 8, 'conv2d spatial preserved with padding=1');

// conv2d stride=2
const conv2dS2 = new Conv2d(3, 8, 3, 2, 0);
const conv2dS2Out = conv2dS2.forward(Tensor.rand([1, 3, 8, 8]));
assert(conv2dS2Out.shape[2] === 3 && conv2dS2Out.shape[3] === 3, 'conv2d stride=2 spatial = (8-3)/2+1 = 3');

// ============================================================
// Pooling
// ============================================================

section('Pooling');

const poolInput = Tensor.fromFloat32(new Float32Array([
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
]), [1, 1, 4, 4]);

// avgpool2d
const avgOut = avgpool2d(poolInput, 2, 2);
assert(avgOut.shape[2] === 2 && avgOut.shape[3] === 2, 'avgpool2d shape');
const avgData = avgOut.toFloat32();
assertClose(avgData[0], (1 + 2 + 5 + 6) / 4, 1e-3, 'avgpool2d top-left');
assertClose(avgData[3], (11 + 12 + 15 + 16) / 4, 1e-3, 'avgpool2d bottom-right');

// maxpool2d
const maxOut = maxpool2d(poolInput, 2, 2);
assert(maxOut.shape[2] === 2 && maxOut.shape[3] === 2, 'maxpool2d shape');
const maxData = maxOut.toFloat32();
assertClose(maxData[0], 6, 1e-3, 'maxpool2d top-left = max(1,2,5,6)');
assertClose(maxData[3], 16, 1e-3, 'maxpool2d bottom-right = max(11,12,15,16)');

// multi-channel pooling
const mcPool = Tensor.rand([2, 3, 6, 6]);
const mcAvg = avgpool2d(mcPool, 2, 2);
assert(mcAvg.shape[0] === 2 && mcAvg.shape[1] === 3, 'avgpool2d preserves batch and channels');
assert(mcAvg.shape[2] === 3 && mcAvg.shape[3] === 3, 'avgpool2d spatial 6/2 = 3');

// ============================================================
// Tile
// ============================================================

section('Tile');

const tileInput = Tensor.fromFloat32(new Float32Array([1, 2, 3]), [1, 3]);
const tiled = tile(tileInput, [2, 1]);
assert(tiled.shape[0] === 2 && tiled.shape[1] === 3, 'tile shape [1,3] x [2,1] -> [2,3]');
const tiledData = tiled.toFloat32();
assertClose(tiledData[0], 1, 1e-6, 'tile row0[0]');
assertClose(tiledData[3], 1, 1e-6, 'tile row1[0] = replicated');
assertClose(tiledData[5], 3, 1e-6, 'tile row1[2] = replicated');

const tiled2 = tile(Tensor.fromFloat32(new Float32Array([1, 2]), [1, 2]), [3, 2]);
assert(tiled2.shape[0] === 3 && tiled2.shape[1] === 4, 'tile [3,2] on [1,2] -> [3,4]');

// ============================================================
// Embedding
// ============================================================

section('Embedding');

const embed = new Embedding(10, 4);
const embedOut = embed.forward([[0, 1, 2], [3, 4, 5]]);
assert(embedOut.shape[0] === 2, 'embedding batch');
assert(embedOut.shape[1] === 3, 'embedding seq_len');
assert(embedOut.shape[2] === 4, 'embedding embed_dim');

// same index should yield same embedding
const embedSame = embed.forward([[0, 0], [0, 0]]);
const esd = embedSame.toFloat32();
assertClose(esd[0], esd[4], 1e-6, 'same index = same embedding[0]');

// embedding has parameters
const embedParams = embed.parameters();
assert(embedParams.length === 1, 'embedding has 1 parameter (weight)');

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

// free
const toFree = Tensor.rand([10]);
toFree.free();
// if free didn't crash, it worked

// data compatibility stub
const dataStub = Tensor.fromFloat32(new Float32Array([1, 2]), [2]);
const storage = dataStub.data.storage;
assert(storage instanceof Float32Array, 'data.storage returns Float32Array');
assertClose(storage[0], 1, 1e-6, 'data.storage values');

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
assertClose(gradX.toFloat32()[0], 6, 1e-3, 'd/dx(x^2) at x=3 = 6');

// gradient through mul
const paramA = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4]), [2, 2]).setRequiresGrad(true);
const paramB = Tensor.fromFloat32(new Float32Array([5, 6, 7, 8]), [2, 2]).setRequiresGrad(true);
const c = paramA.mul(paramB).sum(0).sum(0);
c.backward();
assert(paramA.grad !== null, 'paramA gradient exists');
assert(paramB.grad !== null, 'paramB gradient exists');

// gradient of paramA should be paramB values
const gradAData = paramA.grad.toFloat32();
assertClose(gradAData[0], 5, 1e-3, 'grad_a[0] = b[0]');
assertClose(gradAData[1], 6, 1e-3, 'grad_a[1] = b[1]');

// gradient through add
const addX = Tensor.fromFloat32(new Float32Array([2, 3]), [2]).setRequiresGrad(true);
const addY = Tensor.fromFloat32(new Float32Array([4, 5]), [2]).setRequiresGrad(true);
const addSum = addX.add(addY).sum();
addSum.backward();
assert(addX.grad !== null && addY.grad !== null, 'add gradients exist');
assertClose(addX.grad.toFloat32()[0], 1, 1e-3, 'd/dx(x+y).sum() = 1');
assertClose(addY.grad.toFloat32()[0], 1, 1e-3, 'd/dy(x+y).sum() = 1');

// gradient through matmul
const mmX = Tensor.fromFloat32(new Float32Array([1, 0, 0, 1]), [2, 2]).setRequiresGrad(true);
const mmY = Tensor.fromFloat32(new Float32Array([3, 4, 5, 6]), [2, 2]).setRequiresGrad(true);
const mmOut = mmX.matmul(mmY).sum();
mmOut.backward();
assert(mmX.grad !== null, 'matmul grad exists');

// ============================================================
// Module system
// ============================================================

section('Module system');

class TestNet extends Module {
    constructor() {
        super();
        this.l1 = new Linear(3, 4);
        this.l2 = new Linear(4, 2);
        this.relu = new ReLU();
    }
    forward(x) {
        return this.l2.forward(this.relu.forward(this.l1.forward(x)));
    }
}

const net = new TestNet();

// parameters
const params = net.parameters();
assert(params.length === 4, 'TestNet has 4 parameters (2 weights + 2 biases)');

// namedParameters
const named = net.namedParameters();
assert(named.length === 4, 'namedParameters count');
const names = named.map(([n]) => n);
assert(names.some(n => n.includes('l1')), 'namedParameters includes l1');
assert(names.some(n => n.includes('l2')), 'namedParameters includes l2');

// children
const kids = net.children();
assert(kids.length === 3, 'TestNet has 3 children (l1, l2, relu)');

// modules
const allMods = net.modules();
assert(allMods.length >= 4, 'modules() includes self + children');

// train/eval
net.eval();
assert(net.training === false, 'eval sets training=false');
net.train();
assert(net.training === true, 'train sets training=true');

// forward pass
const netInput = Tensor.rand([2, 3]);
const netOut = net.forward(netInput);
assert(netOut.shape[0] === 2 && netOut.shape[1] === 2, 'TestNet output shape');

// ============================================================
// Optimizers
// ============================================================

section('Optimizers');

// SGD
const sgdParam = Tensor.fromFloat32(new Float32Array([5, 5, 5, 5]), [2, 2]).setRequiresGrad(true);
const sgdParamObj = new Parameter(sgdParam);
const sgdTarget = Tensor.zeros([2, 2]);
const sgdLoss = sgdParamObj.value.sub(sgdTarget).pow(2).mean();
sgdLoss.backward();
const sgd = new SGD([sgdParamObj], 0.1);
const sgdBefore = sgdParamObj.value.toFloat32()[0];
sgd.step();
const sgdAfter = sgdParamObj.value.toFloat32()[0];
assert(sgdAfter < sgdBefore, 'SGD step reduces parameter toward target');
sgd.zeroGrad();

// Adam
const adamParam = Tensor.fromFloat32(new Float32Array([5, 5, 5, 5]), [2, 2]).setRequiresGrad(true);
const adamParamObj = new Parameter(adamParam);
const adamTarget = Tensor.zeros([2, 2]);
const adamLoss = adamParamObj.value.sub(adamTarget).pow(2).mean();
adamLoss.backward();
const adam = new Adam([adamParamObj], { lr: 0.01 });
adam.step();
const adamAfter = adamParamObj.value.toFloat32()[0];
assert(adamAfter !== 5, 'Adam step changes parameter');
adam.zeroGrad();

// Adam returns grad norm
const adamParam2 = Tensor.fromFloat32(new Float32Array([3, 3]), [2]).setRequiresGrad(true);
const adamParamObj2 = new Parameter(adamParam2);
const adamLoss2 = adamParamObj2.value.pow(2).sum();
adamLoss2.backward();
const adam2 = new Adam([adamParamObj2], { lr: 0.01 });
const gradNorm = adam2.step();
assert(typeof gradNorm === 'number', 'Adam.step() returns grad norm');

// GradScaler (basic API tests — unscaleAndStep requires GPU/mixed-precision native support)
const scaler = new GradScaler({ initScale: 1024 });
assert(scaler.getScale() === 1024, 'GradScaler initial scale');

const gsLossInput = Tensor.fromFloat32(new Float32Array([2, 3]), [2]);
const scaledLoss = scaler.scaleLoss(gsLossInput);
const scaledData = scaledLoss.toFloat32();
assertClose(scaledData[0], 2 * 1024, 1e-1, 'scaleLoss scales by initScale');
assertClose(scaledData[1], 3 * 1024, 1e-1, 'scaleLoss scales second element');

// ============================================================
// Dropout
// ============================================================

section('Dropout');

const dropInput = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4]), [4]);

// inference mode: passthrough
const dropInf = dropout(dropInput, 0.5, true);
const dropInfData = dropInf.toFloat32();
assertClose(dropInfData[0], 1, 1e-6, 'dropout inference passthrough[0]');
assertClose(dropInfData[3], 4, 1e-6, 'dropout inference passthrough[3]');

// rate=0: passthrough
const dropZero = dropout(dropInput, 0.0, false);
const dropZeroData = dropZero.toFloat32();
assertClose(dropZeroData[0], 1, 1e-6, 'dropout rate=0 passthrough');

// training mode with rate > 0: some values should be zero
const dropBig = Tensor.ones([1000]);
const dropped = dropout(dropBig, 0.5, false);
const droppedData = dropped.toFloat32();
const zeroCount = droppedData.filter(v => v === 0).length;
assert(zeroCount > 100, 'dropout rate=0.5 zeroes some values');
assert(zeroCount < 900, 'dropout rate=0.5 keeps some values');

// ============================================================
// randRange
// ============================================================

section('randRange');

const rr = randRange([100], -2, 2);
const rrData = rr.toFloat32();
assert(rrData.every(v => v >= -2 && v <= 2), 'randRange within [min, max]');
assert(rr.shape[0] === 100, 'randRange shape');

// ============================================================
// Broadcasting
// ============================================================

section('Broadcasting');

// scalar broadcast
const bcastA = Tensor.fromFloat32(new Float32Array([1, 2, 3]), [3]);
const bcastScalar = Tensor.fromFloat32(new Float32Array([10]), [1]);
const bcastAdd = bcastA.add(bcastScalar).toFloat32();
assertClose(bcastAdd[0], 11, 1e-4, 'broadcast add [3]+[1] [0]');
assertClose(bcastAdd[2], 13, 1e-4, 'broadcast add [3]+[1] [2]');

const bcastMul = bcastA.mul(bcastScalar).toFloat32();
assertClose(bcastMul[0], 10, 1e-4, 'broadcast mul [3]*[1] [0]');
assertClose(bcastMul[2], 30, 1e-4, 'broadcast mul [3]*[1] [2]');

// ============================================================
// Linear module (detailed)
// ============================================================

section('Linear module');

const linear = new Linear(3, 2);
const linInput = Tensor.rand([4, 3]);
const linOutput = linear.forward(linInput);
assert(linOutput.shape[0] === 4 && linOutput.shape[1] === 2, 'linear [4,3] -> [4,2]');

// single sample
const linSingle = linear.forward(Tensor.rand([1, 3]));
assert(linSingle.shape[0] === 1 && linSingle.shape[1] === 2, 'linear single sample');

// parameters
const linParams = linear.parameters();
assert(linParams.length === 2, 'linear has weight + bias');

// ============================================================
// FlashAttention (GPU/CUDA only — skip if not available)
// ============================================================

section('FlashAttention / ResidualLayerNorm / BiasGelu');

if (typeof native.flashAttention === 'function') {
    const nHeads = 2, seqLen = 4, headDim = 8;
    const qAtt = Tensor.rand([1, nHeads, seqLen, headDim]);
    const kAtt = Tensor.rand([1, nHeads, seqLen, headDim]);
    const vAtt = Tensor.rand([1, nHeads, seqLen, headDim]);
    const scale = 1.0 / Math.sqrt(headDim);
    const attOut = flashAttention(qAtt, kAtt, vAtt, scale, true);
    assert(attOut.shape[0] === 1, 'attention batch');
    assert(attOut.shape[1] === nHeads, 'attention heads');
    assert(attOut.shape[2] === seqLen, 'attention seq_len');
    assert(attOut.shape[3] === headDim, 'attention head_dim');
} else {
    console.log('  (skipped — flashAttention not available in CPU build)');
}

if (typeof native.residualLayernorm === 'function') {
    const rlnX = Tensor.rand([2, 4]);
    const rlnResidual = Tensor.rand([2, 4]);
    const rlnGamma = Tensor.ones([4]).setRequiresGrad(true);
    const rlnBeta = Tensor.zeros([4]).setRequiresGrad(true);
    const rlnOut = residualLayerNorm(rlnX, rlnResidual, rlnGamma, rlnBeta);
    assert(rlnOut.shape[0] === 2 && rlnOut.shape[1] === 4, 'residualLayerNorm shape');
} else {
    console.log('  (skipped — residualLayerNorm not available in CPU build)');
}

if (typeof native.biasGelu === 'function') {
    const bgX = Tensor.rand([2, 4]);
    const bgBias = Tensor.rand([4]);
    const bgOut = biasGelu(bgX, bgBias);
    assert(bgOut.shape[0] === 2 && bgOut.shape[1] === 4, 'biasGelu shape');
} else {
    console.log('  (skipped — biasGelu not available in CPU build)');
}

// ============================================================
// End-to-end training loop
// ============================================================

section('End-to-end training');

// Simple regression: learn y = 2x + 1 (deterministic, not sensitive to init)
const trainX = Tensor.fromFloat32(new Float32Array([0, 1, 2, 3, 4, 5]), [6, 1]);
const trainY = Tensor.fromFloat32(new Float32Array([1, 3, 5, 7, 9, 11]), [6, 1]);
const regNet = new Linear(1, 1);
const regOptim = new Adam(regNet.parameters(), { lr: 0.05 });

let earlyLoss = null;
for (let i = 0; i < 200; i++) {
    regOptim.zeroGrad();
    const pred = regNet.forward(trainX);
    const loss = mseLoss(pred, trainY);
    if (i === 10) earlyLoss = loss.toFloat32()[0];
    loss.backward();
    regOptim.step();
}
const finalPred = regNet.forward(trainX);
const finalLoss = mseLoss(finalPred, trainY).toFloat32()[0];
assert(finalLoss < earlyLoss, 'training reduces loss');
assert(finalLoss < 1.0, 'training converges to low loss');

// ============================================================
// Summary
// ============================================================

console.log(`\n${'='.repeat(50)}`);
if (failed === 0) {
    console.log(`All ${passed} tests passed!`);
} else {
    console.log(`${passed} passed, ${failed} FAILED:`);
    for (const f of failures) console.log(`  - ${f}`);
    process.exit(1);
}
