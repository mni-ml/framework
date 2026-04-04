import { Tensor, datasets, SGD, Module, Parameter, Linear, destroyPool } from "@mni-ml/framework";

type Point = [number, number];
type Graph = { N: number; X: Point[]; y: number[] };

// ============================================================
// CLI argument parsing
// ============================================================

interface Args {
    backend: "cpu";
    hidden: number;
    dataset: string;
    rate: number;
    epochs: number;
}

function parseArgs(): Args {
    const args = process.argv.slice(2);
    const parsed: Args = {
        backend: "cpu",
        hidden: 100,
        dataset: "all",
        rate: 0.05,
        epochs: 500,
    };

    for (let i = 0; i < args.length; i++) {
        switch (args[i]) {
            case "--BACKEND":
                parsed.backend = args[++i] as "cpu";
                break;
            case "--HIDDEN":
                parsed.hidden = parseInt(args[++i]!, 10);
                break;
            case "--DATASET":
                parsed.dataset = args[++i]!.toLowerCase();
                break;
            case "--RATE":
                parsed.rate = parseFloat(args[++i]!);
                break;
            case "--EPOCHS":
                parsed.epochs = parseInt(args[++i]!, 10);
                break;
        }
    }

    return parsed;
}

// ============================================================
// Network
// ============================================================

class Network extends Module {
    layer1!: Linear;
    layer2!: Linear;
    layer3!: Linear;

    constructor(hiddenLayers: number) {
        super();
        this.layer1 = new Linear(2, hiddenLayers);
        this.layer2 = new Linear(hiddenLayers, hiddenLayers);
        this.layer3 = new Linear(hiddenLayers, 1);
    }

    forward(x: Tensor): Tensor {
        let h = this.layer1.forward(x).relu();
        h = this.layer2.forward(h).relu();
        return this.layer3.forward(h).sigmoid();
    }
}

// ============================================================
// Training
// ============================================================

function train(
    datasetName: string,
    data: Graph,
    hidden: number,
    learningRate: number,
    maxEpochs: number,
) {
    const model = new Network(hidden);
    const optim = new SGD(model.parameters() as Parameter<Tensor>[], learningRate);

    const X = Tensor.tensor(data.X);
    const y = Tensor.tensor(data.y);

    const epochTimes: number[] = [];
    let finalLoss = 0;
    let finalCorrect = 0;

    console.log(`\n${"=".repeat(60)}`);
    console.log(`Dataset: ${datasetName}  |  Hidden: ${hidden}  |  LR: ${learningRate}  |  Epochs: ${maxEpochs}`);
    console.log(`Backend: cpu (fast_ops parallel element-wise + CPU matmul)`);
    console.log(`${"=".repeat(60)}`);

    for (let epoch = 1; epoch <= maxEpochs; epoch++) {
        const epochStart = performance.now();

        optim.zeroGrad();

        const out = model.forward(X).view(data.N);

        // Binary cross-entropy: -log(p*y + (1-p)*(1-y))
        const prob = out.mul(y).add(out.sub(1.0).mul(y.sub(1.0)));
        const loss = prob.log().neg();

        loss.mul(1 / data.N).sum().backward();
        const totalLoss = loss.sum().item();

        optim.step();

        X.zero_grad_();
        y.zero_grad_();

        const epochEnd = performance.now();
        const epochMs = epochEnd - epochStart;
        epochTimes.push(epochMs);

        if (epoch % 10 === 0 || epoch === maxEpochs || epoch === 1) {
            let correct = 0;
            for (let i = 0; i < data.N; i++) {
                const pred = out.get([i]) > 0.5 ? 1 : 0;
                if (pred === data.y[i]) correct++;
            }
            finalLoss = totalLoss;
            finalCorrect = correct;

            console.log(
                `Epoch ${String(epoch).padStart(4)} | ` +
                `Loss: ${totalLoss.toFixed(4).padStart(10)} | ` +
                `Correct: ${correct}/${data.N} | ` +
                `Time: ${epochMs.toFixed(1)}ms`
            );
        }
    }

    const avgTime = epochTimes.reduce((a, b) => a + b, 0) / epochTimes.length;
    const totalTime = epochTimes.reduce((a, b) => a + b, 0);

    console.log(`\n--- Results for ${datasetName} ---`);
    console.log(`Final loss:       ${finalLoss.toFixed(4)}`);
    console.log(`Final accuracy:   ${finalCorrect}/${data.N}`);
    console.log(`Avg time/epoch:   ${avgTime.toFixed(2)}ms`);
    console.log(`Total time:       ${(totalTime / 1000).toFixed(2)}s`);

    return { datasetName, finalLoss, finalCorrect, total: data.N, avgTime, totalTime };
}

// ============================================================
// Main
// ============================================================

const DATASET_MAP: Record<string, (n: number) => Graph> = {
    simple:  (n) => datasets["Simple"](n) as Graph,
    diag:    (n) => datasets["Diag"](n) as Graph,
    split:   (n) => datasets["Split"](n) as Graph,
    xor:     (n) => datasets["Xor"](n) as Graph,
    circle:  (n) => datasets["Circle"](n) as Graph,
    spiral:  (n) => datasets["Spiral"](n) as Graph,
};

export default function runFastTensor() {
    const args = parseArgs();
    const PTS = 50;

    const datasetNames = args.dataset === "all"
        ? Object.keys(DATASET_MAP)
        : [args.dataset];

    const results: ReturnType<typeof train>[] = [];

    for (const name of datasetNames) {
        const datasetFn = DATASET_MAP[name];
        if (!datasetFn) {
            console.error(`Unknown dataset: ${name}. Available: ${Object.keys(DATASET_MAP).join(", ")}`);
            process.exit(1);
        }

        const data = datasetFn(PTS);
        const epochs = ["circle", "spiral"].includes(name) ? Math.max(args.epochs, 1000) : args.epochs;
        results.push(train(name, data, args.hidden, args.rate, epochs));
    }

    if (results.length > 1) {
        console.log(`\n\n${"=".repeat(60)}`);
        console.log("SUMMARY");
        console.log(`${"=".repeat(60)}`);
        console.log(
            "Dataset".padEnd(10) +
            "Loss".padStart(12) +
            "Accuracy".padStart(12) +
            "Avg ms/epoch".padStart(14) +
            "Total (s)".padStart(12)
        );
        console.log("-".repeat(60));
        for (const r of results) {
            console.log(
                r.datasetName.padEnd(10) +
                r.finalLoss.toFixed(4).padStart(12) +
                `${r.finalCorrect}/${r.total}`.padStart(12) +
                r.avgTime.toFixed(2).padStart(14) +
                (r.totalTime / 1000).toFixed(2).padStart(12)
            );
        }
    }

    destroyPool();
}
