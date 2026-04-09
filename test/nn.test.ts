import {
    Tensor,
    Linear, ReLU, Sigmoid, Tanh, Embedding,
    Conv1d, Conv2d,
    softmax, logsoftmax, gelu, dropout,
    crossEntropyLoss, mseLoss, layerNorm,
    randRange, tile, avgpool2d, maxpool2d,
} from '../dist/index.js';
import { assert, assertClose, section } from './helpers.js';

// ============================================================
// Activations: relu, gelu, modules
// ============================================================

section('Activations');

const actInput = Tensor.fromFloat32(new Float32Array([-2, -1, 0, 1, 2]), [5]);

const reluOut = actInput.relu().toFloat32();
assert(reluOut[0] === 0 && reluOut[1] === 0, 'relu: negatives become 0');
assert(reluOut[2] === 0, 'relu: zero stays 0');
assert(reluOut[3] === 1 && reluOut[4] === 2, 'relu: positives unchanged');

const geluOut = gelu(actInput).toFloat32();
assert(geluOut[0] <= 0, 'gelu(-2) <= 0');
assertClose(geluOut[2], 0, 1e-3, 'gelu(0) ~ 0');
assert(geluOut[4] > 1.5, 'gelu(2) > 1.5');

const reluMod = new ReLU();
const reluModOut = reluMod.forward(actInput).toFloat32();
assert(reluModOut[0] === 0 && reluModOut[4] === 2, 'ReLU module');

const sigMod = new Sigmoid();
const sigModOut = sigMod.forward(Tensor.fromFloat32(new Float32Array([0]), [1])).toFloat32();
assertClose(sigModOut[0], 0.5, 1e-4, 'Sigmoid module(0) = 0.5');

const tanhMod = new Tanh();
const tanhOut = tanhMod.forward(Tensor.fromFloat32(new Float32Array([0]), [1])).toFloat32();
assertClose(tanhOut[0], 0, 1e-3, 'Tanh(0) ~ 0');

// ============================================================
// Softmax / LogSoftmax
// ============================================================

section('Softmax / LogSoftmax');

const smInput = Tensor.fromFloat32(new Float32Array([1, 2, 3, 1, 2, 3]), [2, 3]);
const smOut = softmax(smInput, 1);
const smData = smOut.toFloat32();

assertClose(smData[0] + smData[1] + smData[2], 1.0, 1e-4, 'softmax row0 sums to 1');
assertClose(smData[3] + smData[4] + smData[5], 1.0, 1e-4, 'softmax row1 sums to 1');
assert(smData.every((v: number) => v > 0), 'softmax all positive');
assert(smData[2] > smData[1] && smData[1] > smData[0], 'softmax preserves ordering');

const lsmOut = logsoftmax(smInput, 1).toFloat32();
assert(lsmOut.every((v: number) => v <= 0), 'logsoftmax all <= 0');
const expLsm = Math.exp(lsmOut[0]) + Math.exp(lsmOut[1]) + Math.exp(lsmOut[2]);
assertClose(expLsm, 1.0, 1e-3, 'exp(logsoftmax) sums to 1');

// ============================================================
// Loss functions
// ============================================================

section('Loss functions');

const ceLogits = Tensor.fromFloat32(new Float32Array([2, 1, 0.1, 0.1, 1, 2]), [2, 3]);
const ceTargets = [[0], [2]];
const ceLoss = crossEntropyLoss(ceLogits, ceTargets);
const ceLossVal = ceLoss.toFloat32()[0];
assert(ceLossVal > 0, 'crossEntropyLoss positive');
assert(ceLossVal < 5, 'crossEntropyLoss reasonable range');

const perfectLogits = Tensor.fromFloat32(new Float32Array([100, -100, -100, -100, -100, 100]), [2, 3]);
const perfectLoss = crossEntropyLoss(perfectLogits, [[0], [2]]).toFloat32()[0];
assert(perfectLoss < ceLossVal, 'perfect prediction has lower loss');

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
const row0Mean = (lnData[0] + lnData[1] + lnData[2]) / 3;
assertClose(row0Mean, 0, 1e-3, 'layerNorm row0 zero mean');

const row1Mean = (lnData[3] + lnData[4] + lnData[5]) / 3;
assertClose(row1Mean, 0, 1e-3, 'layerNorm row1 zero mean');

// ============================================================
// Conv1d
// ============================================================

section('Conv1d');

const conv1d = new Conv1d(3, 8, 3, 1, 1);
const conv1dInput = Tensor.rand([2, 3, 10]);
const conv1dOut = conv1d.forward(conv1dInput);
assert(conv1dOut.shape[0] === 2, 'conv1d batch preserved');
assert(conv1dOut.shape[1] === 8, 'conv1d out_channels');
assert(conv1dOut.shape[2] === 10, 'conv1d length preserved with padding=1');

const conv1dS2 = new Conv1d(3, 4, 3, 2, 0);
const conv1dS2Out = conv1dS2.forward(Tensor.rand([1, 3, 10]));
assert(conv1dS2Out.shape[0] === 1 && conv1dS2Out.shape[1] === 4, 'conv1d stride=2 batch/channels');
assert(conv1dS2Out.shape[2] === 4, 'conv1d stride=2 length = (10-3)/2+1 = 4');

const conv1dParams = conv1d.parameters();
assert(conv1dParams.length === 2, 'conv1d has 2 parameters (weight + bias)');

// ============================================================
// Conv2d
// ============================================================

section('Conv2d');

const conv2d = new Conv2d(3, 16, 3, 1, 1);
const conv2dInput = Tensor.rand([1, 3, 8, 8]);
const conv2dOut = conv2d.forward(conv2dInput);
assert(conv2dOut.shape[0] === 1, 'conv2d batch preserved');
assert(conv2dOut.shape[1] === 16, 'conv2d out_channels');
assert(conv2dOut.shape[2] === 8 && conv2dOut.shape[3] === 8, 'conv2d spatial preserved with padding=1');

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

const avgOut = avgpool2d(poolInput, 2, 2);
assert(avgOut.shape[2] === 2 && avgOut.shape[3] === 2, 'avgpool2d shape');
const avgData = avgOut.toFloat32();
assertClose(avgData[0], (1 + 2 + 5 + 6) / 4, 1e-3, 'avgpool2d top-left');
assertClose(avgData[3], (11 + 12 + 15 + 16) / 4, 1e-3, 'avgpool2d bottom-right');

const maxOut = maxpool2d(poolInput, 2, 2);
assert(maxOut.shape[2] === 2 && maxOut.shape[3] === 2, 'maxpool2d shape');
const maxData = maxOut.toFloat32();
assertClose(maxData[0], 6, 1e-3, 'maxpool2d top-left = max(1,2,5,6)');
assertClose(maxData[3], 16, 1e-3, 'maxpool2d bottom-right = max(11,12,15,16)');

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

const embedSame = embed.forward([[0, 0], [0, 0]]);
const esd = embedSame.toFloat32();
assertClose(esd[0], esd[4], 1e-6, 'same index = same embedding[0]');

const embedParams = embed.parameters();
assert(embedParams.length === 1, 'embedding has 1 parameter (weight)');

// ============================================================
// Linear module
// ============================================================

section('Linear module');

const linear = new Linear(3, 2);
const linInput = Tensor.rand([4, 3]);
const linOutput = linear.forward(linInput);
assert(linOutput.shape[0] === 4 && linOutput.shape[1] === 2, 'linear [4,3] -> [4,2]');

const linSingle = linear.forward(Tensor.rand([1, 3]));
assert(linSingle.shape[0] === 1 && linSingle.shape[1] === 2, 'linear single sample');

const linParams = linear.parameters();
assert(linParams.length === 2, 'linear has weight + bias');

// ============================================================
// Dropout
// ============================================================

section('Dropout');

const dropInput = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4]), [4]);

const dropInf = dropout(dropInput, 0.5, true);
const dropInfData = dropInf.toFloat32();
assertClose(dropInfData[0], 1, 1e-6, 'dropout inference passthrough[0]');
assertClose(dropInfData[3], 4, 1e-6, 'dropout inference passthrough[3]');

const dropZero = dropout(dropInput, 0.0, false);
const dropZeroData = dropZero.toFloat32();
assertClose(dropZeroData[0], 1, 1e-6, 'dropout rate=0 passthrough');

const dropBig = Tensor.ones([1000]);
const dropped = dropout(dropBig, 0.5, false);
const droppedData = dropped.toFloat32();
const zeroCount = droppedData.filter((v: number) => v === 0).length;
assert(zeroCount > 100, 'dropout rate=0.5 zeroes some values');
assert(zeroCount < 900, 'dropout rate=0.5 keeps some values');

// ============================================================
// randRange
// ============================================================

section('randRange');

const rr = randRange([100], -2, 2);
const rrData = rr.toFloat32();
assert(rrData.every((v: number) => v >= -2 && v <= 2), 'randRange within [min, max]');
assert(rr.shape[0] === 100, 'randRange shape');

