import { Tensor, destroyPool } from '@mni-ml/framework';
import { pathToFileURL } from 'node:url';

function train(batchSize, features, epochs, lr) {
    const x = Tensor.zeros([batchSize, features]);
    x.data.storage.fill(1);
    const y = Tensor.zeros([batchSize, 1]);

    const weights = Tensor.zeros([features]);
    for (let i = 0; i < features; i++) {
        weights.data.storage[i] = i + 1;
    }

    const invBatch = 1 / batchSize;

    for (let epoch = 0; epoch < epochs; epoch++) {
        const pred = x.mul(weights).sum(1);
        const diff = pred.sub(y);
        const loss = diff.mul(diff).sum(0).mul(invBatch);

        loss.backward();

        const grad = weights.grad;
        if (!grad) {
            throw new Error('Expected gradient on weights');
        }

        for (let i = 0; i < features; i++) {
            weights.data.storage[i] -= lr * grad.data.storage[i];
        }

        weights.zero_grad_();
    }

    return Array.from(weights.data.storage);
}

export function runDispatchParity() {
    const FEATURES = 8;
    const EPOCHS = 3;
    const LEARNING_RATE = 0.01;
    const BELOW_THRESHOLD = 4095;
    const ABOVE_THRESHOLD = 4097;

    try {
        const weightsBelow = train(BELOW_THRESHOLD, FEATURES, EPOCHS, LEARNING_RATE);
        const weightsAbove = train(ABOVE_THRESHOLD, FEATURES, EPOCHS, LEARNING_RATE);

        return { weightsBelow, weightsAbove };
    } finally {
        destroyPool();
    }
}

const isMain =
    import.meta.url === pathToFileURL(process.argv[1] ?? '').href;

if (isMain) {
    const result = runDispatchParity();
    process.stdout.write(JSON.stringify(result));
}
