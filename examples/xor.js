/**
 * XOR example: Train a small neural network to learn XOR.
 *
 * Usage: node examples/xor.js
 */
import { Tensor, Linear, Adam } from '../dist/index.js';

class XORNet {
    constructor() {
        this.fc1 = new Linear(2, 16);
        this.fc2 = new Linear(16, 1);
    }

    forward(x) {
        let h = this.fc1.forward(x).relu();
        return this.fc2.forward(h).sigmoid();
    }

    parameters() {
        return [...this.fc1.parameters(), ...this.fc2.parameters()];
    }
}

const model = new XORNet();
const optimizer = new Adam(model.parameters(), 0.01);

const inputs = Tensor.fromFloat32(new Float32Array([0,0, 0,1, 1,0, 1,1]), [4, 2]);
const targets = Tensor.fromFloat32(new Float32Array([0, 1, 1, 0]), [4, 1]);

console.log('Training XOR network...\n');

for (let epoch = 0; epoch < 1000; epoch++) {
    const pred = model.forward(inputs);
    const diff = pred.sub(targets);
    const loss = diff.pow(2).mean(0).mean(0);

    loss.backward();
    optimizer.step();
    optimizer.zeroGrad();

    if (epoch % 100 === 0) {
        const lossVal = loss.toFloat32()[0];
        console.log(`Epoch ${epoch}: loss = ${lossVal.toFixed(6)}`);
    }
}

console.log('\nFinal predictions:');
const finalPred = model.forward(inputs).toFloat32();
console.log(`  0 XOR 0 = ${finalPred[0].toFixed(4)} (expected: 0)`);
console.log(`  0 XOR 1 = ${finalPred[1].toFixed(4)} (expected: 1)`);
console.log(`  1 XOR 0 = ${finalPred[2].toFixed(4)} (expected: 1)`);
console.log(`  1 XOR 1 = ${finalPred[3].toFixed(4)} (expected: 0)`);
