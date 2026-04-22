// End-to-end smoke test: drives the cuTile backend through its N-API
// entry points to confirm the TypeScript-facing surface works.

import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const cutile = require('./mni-framework-cutile.linux-x64-gnu.node');

const approxEq = (a, b, tol = 1e-3) =>
    Math.abs(a - b) <= tol + tol * Math.max(Math.abs(a), Math.abs(b));

function assertEqArr(got, want, tol, label) {
    if (got.length !== want.length) throw new Error(`${label}: length ${got.length} vs ${want.length}`);
    let bad = 0;
    for (let i = 0; i < got.length; ++i) {
        if (!approxEq(got[i], want[i], tol)) {
            if (bad < 3) console.error(`${label}: [${i}] got=${got[i]} want=${want[i]}`);
            bad += 1;
        }
    }
    if (bad > 0) throw new Error(`${label}: ${bad}/${got.length} elements mismatched`);
    console.log(`${label}: OK (${got.length} elements)`);
}

// --- Basic creation / readback ---
const zId = cutile.zeros([4]);
const oId = cutile.ones([4]);
assertEqArr(cutile.toFloat32(zId), new Float32Array([0, 0, 0, 0]), 1e-7, 'zeros');
assertEqArr(cutile.toFloat32(oId), new Float32Array([1, 1, 1, 1]), 1e-7, 'ones');
cutile.freeTensor(zId);
cutile.freeTensor(oId);

// --- add ---
const aArr = Float32Array.from({ length: 1024 }, (_, i) => i * 0.5);
const bArr = Float32Array.from({ length: 1024 }, (_, i) => -i * 0.25);
const aId = cutile.fromFloat32(aArr, [1024]);
const bId = cutile.fromFloat32(bArr, [1024]);
const sumId = cutile.add(aId, bId);
const sumOut = cutile.toFloat32(sumId);
const sumExpect = new Float32Array(1024);
for (let i = 0; i < 1024; ++i) sumExpect[i] = aArr[i] + bArr[i];
assertEqArr(sumOut, sumExpect, 1e-4, 'add');

// --- saxpy (fused) ---
const saxId = cutile.saxpy(3.5, aId, bId);
const saxOut = cutile.toFloat32(saxId);
const saxExpect = new Float32Array(1024);
for (let i = 0; i < 1024; ++i) saxExpect[i] = 3.5 * aArr[i] + bArr[i];
assertEqArr(saxOut, saxExpect, 1e-3, 'saxpy');

// --- relu ---
const rArr = Float32Array.from({ length: 256 }, (_, i) => i * 0.5 - 64);
const rId = cutile.fromFloat32(rArr, [256]);
const rOutId = cutile.relu(rId);
const rOut = cutile.toFloat32(rOutId);
const rExpect = new Float32Array(rArr.length);
for (let i = 0; i < rArr.length; ++i) rExpect[i] = Math.max(0, rArr[i]);
assertEqArr(rOut, rExpect, 1e-6, 'relu');

// --- matmul 16x24 @ 24x32 ---
const M = 16, K = 24, N = 32;
const mA = Float32Array.from({ length: M * K }, (_, i) => (i % 11) * 0.1);
const mB = Float32Array.from({ length: K * N }, (_, i) => (i % 7) * 0.15 - 0.3);
const mAId = cutile.fromFloat32(mA, [M, K]);
const mBId = cutile.fromFloat32(mB, [K, N]);
const mCId = cutile.matmul(mAId, mBId);
const mC = cutile.toFloat32(mCId);
const mExpect = new Float32Array(M * N);
for (let i = 0; i < M; ++i)
    for (let j = 0; j < N; ++j) {
        let acc = 0;
        for (let k = 0; k < K; ++k) acc += mA[i * K + k] * mB[k * N + j];
        mExpect[i * N + j] = acc;
    }
assertEqArr(mC, mExpect, 2e-3, `matmul ${M}x${K}x${N}`);
console.log('matmul output shape:', cutile.tensorShape(mCId));

// --- sum_all ---
const sArr = Float32Array.from({ length: 512 }, (_, i) => i * 0.1);
const sId = cutile.fromFloat32(sArr, [16, 32]);
const totId = cutile.sumAll(sId);
const tot = cutile.getScalar(totId);
let sWant = 0;
for (const v of sArr) sWant += v;
if (!approxEq(tot, sWant, 1e-3)) throw new Error(`sumAll: got ${tot} want ${sWant}`);
console.log(`sumAll: OK (got ${tot.toFixed(3)} ≈ ${sWant.toFixed(3)})`);

console.log('\nAll cuTile napi smoke tests passed.');
