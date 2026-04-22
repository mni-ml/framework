// Verify the framework's native-loader can find and load the cuTile addon.
import { loadCutile } from '../../dist/native-loader.js';

const m = loadCutile();
console.log('loadCutile() returned', typeof m, '— exports:', Object.keys(m).length);

// Smoke check: run a small matmul via the loader-returned module.
const a = m.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
const b = m.fromFloat32(new Float32Array([1, 0, 0, 1, 1, 1]), [3, 2]);
const c = m.matmul(a, b);
const out = m.toFloat32(c);
console.log('[ 2x3 @ 3x2 ]', Array.from(out));
const want = [1 + 0 + 3, 0 + 2 + 3, 4 + 0 + 6, 0 + 5 + 6];
for (let i = 0; i < want.length; ++i) {
    if (Math.abs(out[i] - want[i]) > 1e-3) throw new Error(`matmul mismatch at ${i}: ${out[i]} vs ${want[i]}`);
}
console.log('loader integration OK.');
