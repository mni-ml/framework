import {
    Tensor, TensorData, Module, Parameter, SGD,
    Linear, Conv2dModule, crossEntropyLoss, maxpool2d, dropout, destroyPool,
} from "@mni-ml/framework";
import { loadMnist } from "./mnistLoader.js";
import { writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const NUM_CLASSES = 10;

// ============================================================
// Network: single Conv2d layer to keep runtime feasible
// ============================================================

class MnistCNN extends Module {
    conv1!: Conv2dModule;
    fc1!: Linear;
    fc2!: Linear;

    constructor() {
        super();
        // Input: [B, 1, 14, 14]  (images downsampled from 28x28)
        // Conv2d(1→4, 3x3): [B, 4, 14, 14]
        // MaxPool2d([2,2]): [B, 4, 7, 7]
        // Flatten: [B, 196]
        // FC: 196 → 32 → 10
        this.conv1 = new Conv2dModule(1, 4, [3, 3]);
        this.fc1 = new Linear(4 * 7 * 7, 32);
        this.fc2 = new Linear(32, NUM_CLASSES);
    }

    forward(x: Tensor, train: boolean = true): Tensor {
        let h = this.conv1.forward(x).relu();
        h = maxpool2d(h, [2, 2]);

        const batch = h.shape[0]!;
        h = h.contiguous().view(batch, 4 * 7 * 7);

        h = this.fc1.forward(h).relu();
        h = dropout(h, 0.25, !train);
        return this.fc2.forward(h);
    }
}

// ============================================================
// Helpers
// ============================================================

function argmax(t: Tensor, row: number, cols: number): number {
    let best = 0;
    let bestVal = -Infinity;
    for (let c = 0; c < cols; c++) {
        const v = t.get([row, c]);
        if (v > bestVal) { bestVal = v; best = c; }
    }
    return best;
}

/**
 * Downsample 28x28 → 14x14 by averaging 2x2 blocks.
 */
function downsample(images: Float64Array, count: number): Float64Array {
    const out = new Float64Array(count * 14 * 14);
    for (let n = 0; n < count; n++) {
        const srcBase = n * 28 * 28;
        const dstBase = n * 14 * 14;
        for (let r = 0; r < 14; r++) {
            for (let c = 0; c < 14; c++) {
                const r2 = r * 2, c2 = c * 2;
                const avg = (
                    images[srcBase + r2 * 28 + c2]! +
                    images[srcBase + r2 * 28 + c2 + 1]! +
                    images[srcBase + (r2 + 1) * 28 + c2]! +
                    images[srcBase + (r2 + 1) * 28 + c2 + 1]!
                ) / 4;
                out[dstBase + r * 14 + c] = avg;
            }
        }
    }
    return out;
}

// ============================================================
// Training
// ============================================================

export default async function runMnist() {
    const __dirname = dirname(fileURLToPath(import.meta.url));
    const cacheDir = join(__dirname, "..", "data", "mnist");

    console.log("Loading MNIST data...");
    const data = await loadMnist(cacheDir, 500, 100);
    console.log(`Train: ${data.numTrain}, Test: ${data.numTest}, Image: ${data.rows}x${data.cols}`);

    console.log("Downsampling 28x28 → 14x14...");
    const trainImgs = downsample(data.trainImages, data.numTrain);
    const testImgs = downsample(data.testImages, data.numTest);
    const H = 14, W = 14, imgSize = H * W;

    const model = new MnistCNN();
    const optim = new SGD(model.parameters() as Parameter<Tensor>[], 0.01);

    const BATCH = 4;
    const EPOCHS = 15;
    const logLines: string[] = [];

    function log(msg: string) {
        console.log(msg);
        logLines.push(msg);
    }

    log(`MNIST CNN Training — ${data.numTrain} train, ${data.numTest} test (14x14 downsampled)`);
    log(`Batch: ${BATCH}, Epochs: ${EPOCHS}, LR: 0.01`);
    log("=".repeat(70));

    for (let epoch = 1; epoch <= EPOCHS; epoch++) {
        const epochStart = performance.now();
        model.train();
        let totalLoss = 0;
        let trainCorrect = 0;
        let batches = 0;

        for (let start = 0; start + BATCH <= data.numTrain; start += BATCH) {
            optim.zeroGrad();

            const imgStorage = new Float64Array(BATCH * imgSize);
            const targetStorage = new Float64Array(BATCH * NUM_CLASSES);

            for (let i = 0; i < BATCH; i++) {
                const srcOffset = (start + i) * imgSize;
                imgStorage.set(trainImgs.subarray(srcOffset, srcOffset + imgSize), i * imgSize);
                targetStorage[(i * NUM_CLASSES) + data.trainLabels[start + i]!] = 1;
            }

            const images = new Tensor(new TensorData(imgStorage, [BATCH, 1, H, W]));
            const target = new Tensor(new TensorData(targetStorage, [BATCH, NUM_CLASSES]));

            const output = model.forward(images, true);
            const loss = crossEntropyLoss(output, target);

            loss.backward();
            totalLoss += loss.item();
            batches++;

            for (let i = 0; i < BATCH; i++) {
                if (argmax(output, i, NUM_CLASSES) === data.trainLabels[start + i]) {
                    trainCorrect++;
                }
            }

            optim.step();
        }

        // Validation
        model.eval();
        let valCorrect = 0;
        for (let start = 0; start + BATCH <= data.numTest; start += BATCH) {
            const imgStorage = new Float64Array(BATCH * imgSize);
            for (let i = 0; i < BATCH; i++) {
                const srcOffset = (start + i) * imgSize;
                imgStorage.set(testImgs.subarray(srcOffset, srcOffset + imgSize), i * imgSize);
            }
            const images = new Tensor(new TensorData(imgStorage, [BATCH, 1, H, W]));
            const output = model.forward(images, false);
            for (let i = 0; i < BATCH; i++) {
                if (argmax(output, i, NUM_CLASSES) === data.testLabels[start + i]) {
                    valCorrect++;
                }
            }
        }

        const valTotal = Math.floor(data.numTest / BATCH) * BATCH;
        const trainTotal = Math.floor(data.numTrain / BATCH) * BATCH;
        const epochMs = performance.now() - epochStart;

        log(
            `Epoch ${String(epoch).padStart(2)} | ` +
            `Train Loss: ${(totalLoss / batches).toFixed(4).padStart(8)} | ` +
            `Train Acc: ${trainCorrect}/${trainTotal} (${(100 * trainCorrect / trainTotal).toFixed(1)}%) | ` +
            `Val Acc: ${valCorrect}/${valTotal} (${(100 * valCorrect / valTotal).toFixed(1)}%) | ` +
            `Time: ${(epochMs / 1000).toFixed(1)}s`
        );
    }

    const outPath = join(__dirname, "..", "mnist.txt");
    writeFileSync(outPath, logLines.join("\n") + "\n");
    log(`\nResults saved to ${outPath}`);

    destroyPool();
}
