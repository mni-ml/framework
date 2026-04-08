import { Tensor } from '../toy/tensor.ts';
import { TensorHistory } from '../toy/tensor_functions.ts';

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
// sin forward
// ============================================================

section('sin');

{
    const x = Tensor.tensor([0, Math.PI / 6, Math.PI / 4, Math.PI / 2, Math.PI]);
    const y = x.sin();
    assertClose(y.get([0]), Math.sin(0), 1e-6, 'sin(0)');
    assertClose(y.get([1]), Math.sin(Math.PI / 6), 1e-6, 'sin(pi/6)');
    assertClose(y.get([2]), Math.sin(Math.PI / 4), 1e-6, 'sin(pi/4)');
    assertClose(y.get([3]), Math.sin(Math.PI / 2), 1e-6, 'sin(pi/2)');
    assertClose(y.get([4]), Math.sin(Math.PI), 1e-6, 'sin(pi)');
}

// ============================================================
// cos forward
// ============================================================

section('cos');

{
    const x = Tensor.tensor([0, Math.PI / 6, Math.PI / 4, Math.PI / 2, Math.PI]);
    const y = x.cos();
    assertClose(y.get([0]), Math.cos(0), 1e-6, 'cos(0)');
    assertClose(y.get([1]), Math.cos(Math.PI / 6), 1e-6, 'cos(pi/6)');
    assertClose(y.get([2]), Math.cos(Math.PI / 4), 1e-6, 'cos(pi/4)');
    assertClose(y.get([3]), Math.cos(Math.PI / 2), 1e-6, 'cos(pi/2)');
    assertClose(y.get([4]), Math.cos(Math.PI), 1e-6, 'cos(pi)');
}

// ============================================================
// sqrt forward
// ============================================================

section('sqrt');

{
    const x = Tensor.tensor([0, 1, 4, 9, 16]);
    const y = x.sqrt();
    assertClose(y.get([0]), 0, 1e-4, 'sqrt(0)');
    assertClose(y.get([1]), 1, 1e-6, 'sqrt(1)');
    assertClose(y.get([2]), 2, 1e-6, 'sqrt(4)');
    assertClose(y.get([3]), 3, 1e-6, 'sqrt(9)');
    assertClose(y.get([4]), 4, 1e-6, 'sqrt(16)');
}

// ============================================================
// sin backward
// ============================================================

section('sin backward');

{
    // d/dx sin(x) = cos(x)
    const x = Tensor.tensor([0.5, 1.0, 2.0]);
    x.history = new TensorHistory();
    const y = x.sin().sum();
    y.backward();
    assertClose(x.grad.get([0]), Math.cos(0.5), 1e-5, 'dsin/dx at 0.5');
    assertClose(x.grad.get([1]), Math.cos(1.0), 1e-5, 'dsin/dx at 1.0');
    assertClose(x.grad.get([2]), Math.cos(2.0), 1e-5, 'dsin/dx at 2.0');
}

// ============================================================
// cos backward
// ============================================================

section('cos backward');

{
    // d/dx cos(x) = -sin(x)
    const x = Tensor.tensor([0.5, 1.0, 2.0]);
    x.history = new TensorHistory();
    const y = x.cos().sum();
    y.backward();
    assertClose(x.grad.get([0]), -Math.sin(0.5), 1e-5, 'dcos/dx at 0.5');
    assertClose(x.grad.get([1]), -Math.sin(1.0), 1e-5, 'dcos/dx at 1.0');
    assertClose(x.grad.get([2]), -Math.sin(2.0), 1e-5, 'dcos/dx at 2.0');
}

// ============================================================
// sqrt backward
// ============================================================

section('sqrt backward');

{
    // d/dx sqrt(x) = 1 / (2 * sqrt(x))
    const x = Tensor.tensor([1.0, 4.0, 9.0]);
    x.history = new TensorHistory();
    const y = x.sqrt().sum();
    y.backward();
    assertClose(x.grad.get([0]), 1 / (2 * Math.sqrt(1.0)), 1e-5, 'dsqrt/dx at 1');
    assertClose(x.grad.get([1]), 1 / (2 * Math.sqrt(4.0)), 1e-5, 'dsqrt/dx at 4');
    assertClose(x.grad.get([2]), 1 / (2 * Math.sqrt(9.0)), 1e-5, 'dsqrt/dx at 9');
}

// ============================================================
// gradient chain: sin(cos(x))
// ============================================================

section('gradient chain: sin(cos(x))');

{
    // d/dx sin(cos(x)) = cos(cos(x)) * (-sin(x))
    const x = Tensor.tensor([0.7]);
    x.history = new TensorHistory();
    const y = x.cos().sin().sum();
    y.backward();
    const expected = Math.cos(Math.cos(0.7)) * (-Math.sin(0.7));
    assertClose(x.grad.get([0]), expected, 1e-5, 'chain rule sin(cos(x))');
}

// ============================================================
// finite difference gradient check
// ============================================================

section('finite difference gradient check');

{
    const eps = 1e-5;

    // Check sin gradient via finite differences
    const xVal = 1.3;
    const sinGradAnalytic = Math.cos(xVal);
    const sinGradNumeric = (Math.sin(xVal + eps) - Math.sin(xVal - eps)) / (2 * eps);
    assertClose(sinGradAnalytic, sinGradNumeric, 1e-5, 'sin finite diff');

    // Check cos gradient via finite differences
    const cosGradAnalytic = -Math.sin(xVal);
    const cosGradNumeric = (Math.cos(xVal + eps) - Math.cos(xVal - eps)) / (2 * eps);
    assertClose(cosGradAnalytic, cosGradNumeric, 1e-5, 'cos finite diff');

    // Check sqrt gradient via finite differences on autograd
    const x = Tensor.tensor([2.0]);
    x.history = new TensorHistory();
    x.sqrt().sum().backward();
    const sqrtGradNumeric = (Math.sqrt(2.0 + eps) - Math.sqrt(2.0 - eps)) / (2 * eps);
    assertClose(x.grad.get([0]), sqrtGradNumeric, 1e-4, 'sqrt autograd vs finite diff');
}

// ============================================================
// 2D tensor operations
// ============================================================

section('2D sin/cos/sqrt');

{
    const x = Tensor.tensor([[1, 2], [3, 4]]);
    const s = x.sin();
    assert(s.shape[0] === 2 && s.shape[1] === 2, 'sin preserves shape');
    assertClose(s.get([0, 0]), Math.sin(1), 1e-6, 'sin 2D element');
    assertClose(s.get([1, 1]), Math.sin(4), 1e-6, 'sin 2D element [1,1]');

    const c = x.cos();
    assertClose(c.get([0, 1]), Math.cos(2), 1e-6, 'cos 2D element');

    const q = Tensor.tensor([[1, 4], [9, 16]]).sqrt();
    assertClose(q.get([0, 0]), 1, 1e-6, 'sqrt 2D element [0,0]');
    assertClose(q.get([1, 1]), 4, 1e-6, 'sqrt 2D element [1,1]');
}

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
