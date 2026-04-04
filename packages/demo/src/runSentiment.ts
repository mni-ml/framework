import {
    Tensor, TensorData, Module, Parameter, SGD,
    Linear, Conv1dModule, Embedding, crossEntropyLoss, destroyPool,
} from "@mni-ml/framework";
import { readFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const NUM_CLASSES = 2;

// ============================================================
// Data loading
// ============================================================

interface Sst2Sample { tokens: number[]; label: number; }

interface Sst2Data {
    vocab_size: number;
    embed_dim: number;
    seq_len: number;
    embeddings: number[][];
    train: Sst2Sample[];
    valid: Sst2Sample[];
}

function loadSst2(dataDir: string): Sst2Data {
    const raw = readFileSync(join(dataDir, "sst2.json"), "utf-8");
    return JSON.parse(raw) as Sst2Data;
}

// ============================================================
// Minimal Conv1d sentiment classifier
// Embedding → Conv1d → ReLU → sum pool → Linear
// ============================================================

class SentimentCNN extends Module {
    embedding!: Embedding;
    conv1!: Conv1dModule;
    fc!: Linear;

    constructor(vocabSize: number, embedDim: number) {
        super();
        this.embedding = new Embedding(vocabSize, embedDim);
        this.conv1 = new Conv1dModule(embedDim, 8, 3);
        this.fc = new Linear(8, NUM_CLASSES);
    }

    forward(indices: number[][]): Tensor {
        let h = this.embedding.forward(indices);

        // [B, seqLen, embedDim] → [B, embedDim, seqLen]
        h = h.permute(0, 2, 1).contiguous();

        h = this.conv1.forward(h).relu();

        // Sum pool (not mean — preserves gradient magnitude): [B, 8, seqLen] → [B, 8]
        const batch = indices.length;
        h = h.sum(2).view(batch, 8);

        return this.fc.forward(h);
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

// ============================================================
// Training
// ============================================================

export default function runSentiment() {
    const __dirname = dirname(fileURLToPath(import.meta.url));
    const dataDir = join(__dirname, "..", "data");

    console.log("Loading SST-2 data...");
    const data = loadSst2(dataDir);
    console.log(`Vocab: ${data.vocab_size}, Embed dim: ${data.embed_dim}, Seq len: ${data.seq_len}`);
    console.log(`Train: ${data.train.length}, Valid: ${data.valid.length}`);

    const model = new SentimentCNN(data.vocab_size, data.embed_dim);
    const params = [...model.parameters()] as Parameter<Tensor>[];
    console.log(`Model parameters: ${params.length}`);
    const LR = 0.01;
    const optim = new SGD(params, LR);

    const BATCH = 16;
    const EPOCHS = 30;
    const logLines: string[] = [];

    function log(msg: string) {
        console.log(msg);
        logLines.push(msg);
    }

    log(`SST-2 Sentiment CNN Training — ${data.train.length} train, ${data.valid.length} valid`);
    log(`Batch: ${BATCH}, Epochs: ${EPOCHS}, LR: ${LR}`);
    log("=".repeat(70));

    let bestValAcc = 0;

    for (let epoch = 1; epoch <= EPOCHS; epoch++) {
        const epochStart = performance.now();
        model.train();
        let totalLoss = 0;
        let trainCorrect = 0;
        let batches = 0;

        for (let start = 0; start + BATCH <= data.train.length; start += BATCH) {
            optim.zeroGrad();

            const batchIndices: number[][] = [];
            const targetStorage = new Float64Array(BATCH * NUM_CLASSES);

            for (let i = 0; i < BATCH; i++) {
                const sample = data.train[start + i]!;
                batchIndices.push(sample.tokens);
                targetStorage[i * NUM_CLASSES + sample.label] = 1;
            }

            const target = new Tensor(new TensorData(targetStorage, [BATCH, NUM_CLASSES]));
            const output = model.forward(batchIndices);
            const loss = crossEntropyLoss(output, target);

            loss.backward();
            totalLoss += loss.item();
            batches++;

            for (let i = 0; i < BATCH; i++) {
                if (argmax(output, i, NUM_CLASSES) === data.train[start + i]!.label) {
                    trainCorrect++;
                }
            }

            optim.step();
        }

        // Validation
        model.eval();
        let valCorrect = 0;
        let valTotal = 0;
        for (let start = 0; start + BATCH <= data.valid.length; start += BATCH) {
            const batchIndices: number[][] = [];
            for (let i = 0; i < BATCH; i++) {
                batchIndices.push(data.valid[start + i]!.tokens);
            }
            const output = model.forward(batchIndices);
            for (let i = 0; i < BATCH; i++) {
                if (argmax(output, i, NUM_CLASSES) === data.valid[start + i]!.label) {
                    valCorrect++;
                }
            }
            valTotal += BATCH;
        }

        const trainTotal = Math.floor(data.train.length / BATCH) * BATCH;
        const trainAcc = 100 * trainCorrect / trainTotal;
        const valAcc = 100 * valCorrect / valTotal;
        if (valAcc > bestValAcc) bestValAcc = valAcc;
        const epochMs = performance.now() - epochStart;

        log(
            `Epoch ${String(epoch).padStart(2)} | ` +
            `Train Loss: ${(totalLoss / batches).toFixed(4).padStart(8)} | ` +
            `Train Acc: ${trainCorrect}/${trainTotal} (${trainAcc.toFixed(1)}%) | ` +
            `Val Acc: ${valCorrect}/${valTotal} (${valAcc.toFixed(1)}%) | ` +
            `Time: ${(epochMs / 1000).toFixed(1)}s`
        );
    }

    log(`\nBest validation accuracy: ${bestValAcc.toFixed(1)}%`);

    const outPath = join(__dirname, "..", "sentiment.txt");
    writeFileSync(outPath, logLines.join("\n") + "\n");
    log(`Results saved to ${outPath}`);

    destroyPool();
}
