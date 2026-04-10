import {
    Tensor, native,
    flashAttention, residualLayerNorm, biasGelu,
} from '../dist/index.js';
import { assert, skip, section } from './helpers.js';

// ============================================================
// FlashAttention / ResidualLayerNorm / BiasGelu (GPU/CUDA only)
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
    skip('flashAttention not available in CPU build');
}

if (typeof native.residualLayernorm === 'function') {
    const rlnX = Tensor.rand([2, 4]);
    const rlnResidual = Tensor.rand([2, 4]);
    const rlnGamma = Tensor.ones([4]).setRequiresGrad(true);
    const rlnBeta = Tensor.zeros([4]).setRequiresGrad(true);
    const rlnOut = residualLayerNorm(rlnX, rlnResidual, rlnGamma, rlnBeta);
    assert(rlnOut.shape[0] === 2 && rlnOut.shape[1] === 4, 'residualLayerNorm shape');
} else {
    skip('residualLayerNorm not available in CPU build');
}

if (typeof native.biasGelu === 'function') {
    const bgX = Tensor.rand([2, 4]);
    const bgBias = Tensor.rand([4]);
    const bgOut = biasGelu(bgX, bgBias);
    assert(bgOut.shape[0] === 2 && bgOut.shape[1] === 4, 'biasGelu shape');
} else {
    skip('biasGelu not available in CPU build');
}

