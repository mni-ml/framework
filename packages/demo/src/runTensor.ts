import { Tensor, datasets, SGD, Module, Parameter, Linear, destroyPool } from "@mni-ml/framework";

type Point = [number, number];
type Graph = { N: number; X: Point[]; y: number[] };

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

function defaultLogFn(epoch: number, totalLoss: number, correct: number) {
    console.log("Epoch ", epoch, " loss ", totalLoss, " correct ", correct);
}

class TensorTrain {
    hiddenLayers: number;
    model: Network;
    learningRate!: number;
    maxEpochs!: number;

    constructor(hiddenLayers: number) {
        this.hiddenLayers = hiddenLayers;
        this.model = new Network(hiddenLayers);
    }

    runOne(x: Point): Tensor {
        return this.model.forward(Tensor.tensor([x]));
    }

    train(data: Graph, learningRate: number, maxEpochs: number = 500, logFn = defaultLogFn) {
        this.learningRate = learningRate;
        this.maxEpochs = maxEpochs;
        this.model = new Network(this.hiddenLayers);
        const optim = new SGD(this.model.parameters() as Parameter<Tensor>[], learningRate);

        const X = Tensor.tensor(data.X);
        const y = Tensor.tensor(data.y);

        for (let epoch = 1; epoch <= this.maxEpochs; epoch++) {
            optim.zeroGrad();

            const out = this.model.forward(X).view(data.N);

            // Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
            // Rearranged as: -log(p*y + (1-p)*(1-y))
            const prob = out.mul(y).add(out.sub(1.0).mul(y.sub(1.0)));
            const loss = prob.log().neg();

            loss.mul(1 / data.N).sum().backward();
            const totalLoss = loss.sum().item();

            optim.step();

            X.zero_grad_();
            y.zero_grad_();

            if (epoch % 500 === 0 || epoch === this.maxEpochs) {
                let correct = 0;
                for (let i = 0; i < data.N; i++) {
                    const pred = out.get([i]) > 0.5 ? 1 : 0;
                    if (pred === data.y[i]) correct++;
                }
                logFn(epoch, totalLoss, correct);
            }
        }
    }
}

export default function runTensor() {
    const PTS = 50;

    const data1 = datasets["Simple"](PTS) as Graph;
    const data2 = datasets["Diag"](PTS) as Graph;
    const data3 = datasets["Split"](PTS) as Graph;
    const data4 = datasets["Xor"](PTS) as Graph;
    const data5 = datasets["Circle"](PTS) as Graph;
    const data6 = datasets["Spiral"](PTS) as Graph;

    console.log("=== Simple [4] ===");
    new TensorTrain(4).train(data1, 0.5);

    console.log("\n=== Diag [4] ===");
    new TensorTrain(4).train(data2, 0.5);

    console.log("\n=== Split [8] ===");
    new TensorTrain(8).train(data3, 0.5);

    console.log("\n=== Xor [8] ===");
    new TensorTrain(8).train(data4, 0.5);

    console.log("\n=== Circle [8] ===");
    new TensorTrain(8).train(data5, 0.5, 1000);

    console.log("\n=== Spiral [8] ===");
    new TensorTrain(8).train(data6, 0.5, 1000);

    destroyPool();
}
