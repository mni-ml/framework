import { Scalar, datasets, SGD, Module, Parameter } from "@mni-ml/framework";

type Point = [number, number];
type Graph = { N: number; X: Point[]; y: number[] };

class ScalarLinear extends Module<Parameter<Scalar>> {
  inSize: number;
  outSize: number;
  weights: Parameter<Scalar>[][];
  bias: Parameter<Scalar>[];

  constructor(inSize: number, outSize: number) {
    super();
    this.inSize = inSize;
    this.outSize = outSize;

    this.weights = Array.from({ length: outSize }, () =>
      Array.from({ length: inSize }, () => new Parameter(new Scalar(2 * (Math.random() - 0.5))))
    );

    this.bias = Array.from({ length: outSize }, () => new Parameter(new Scalar(2 * (Math.random() - 0.5))));

    // Register weights and bias as parameters via the Proxy.
    // Array elements aren't auto-captured, so we assign each one
    // to a named property the Proxy can intercept.
    for (let i = 0; i < outSize; i++) {
      (this as any)[`b_${i}`] = this.bias[i];
      for (let j = 0; j < inSize; j++) {
        (this as any)[`w_${i}_${j}`] = this.weights[i]![j];
      }
    }
  }

  forward(inputs: Scalar[]): Scalar[] {
    const outputs: Scalar[] = [];

    for (let i = 0; i < this.outSize; ++i) {
      let result = this.bias[i]!.value.add(0);

      for (let j = 0; j < this.inSize; ++j) {
        result = result.add(this.weights[i]![j]!.value.mul(inputs[j]!));
      }
      outputs.push(result);
    }

    return outputs;
  }
}

class Network extends Module<Parameter<Scalar>> {
  layer1!: ScalarLinear;
  layer2!: ScalarLinear;

  constructor(hiddenLayers: number) {
    super();
    this.layer1 = new ScalarLinear(2, hiddenLayers);
    this.layer2 = new ScalarLinear(hiddenLayers, 1);
  }

  forward(x: [Scalar, Scalar]): Scalar {
    const hidden = this.layer1.forward(x).map(s => s.relu());
    return this.layer2.forward(hidden)[0]!;
  }
}

function defaultLogFn(epoch: number, totalLoss: number, correct: number) {
  console.log("Epoch ", epoch, " loss ", totalLoss, " correct ", correct);
}

class ScalarTrain {
  hiddenLayers: number;
  model: Network;
  learningRate!: number;
  maxEpochs!: number;

  constructor(hiddenLayers: number) {
    this.hiddenLayers = hiddenLayers;
    this.model = new Network(hiddenLayers);
  }

  runOne(x: Point) {
    return this.model.forward([new Scalar(x[0]), new Scalar(x[1])]);
  }

  train(data: Graph, learningRate: number, maxEpochs: number = 500, logFn = defaultLogFn) {
    this.learningRate = learningRate;
    this.maxEpochs = maxEpochs;
    this.model = new Network(this.hiddenLayers);
    const optim = new SGD(this.model.parameters(), learningRate);

    for (let epoch = 1; epoch < this.maxEpochs + 1; ++epoch) {
      let totalLoss = new Scalar(0);
      let correct = 0;

      optim.zeroGrad();

      for (let i = 0; i < data.N; ++i) {
        const [rx1, rx2] = data.X[i]!;
        const y = data.y[i]!;
        const x = this.model.forward([new Scalar(rx1!), new Scalar(rx2!)]);

        if ((x.data > 0 ? 1 : 0) === y) correct++;

        // Binary cross-entropy: log(1 + exp(∓x)) depending on label
        const loss = y === 1
          ? x.neg().exp().add(1).log()
          : x.exp().add(1).log();

        totalLoss = totalLoss.add(loss.div(data.N));
      }

      totalLoss.backward();
      optim.step();

      if (epoch % 500 === 0 || epoch === maxEpochs) {
        logFn(epoch, totalLoss.data, correct);
      }
    }
  }
}

export default function runScalar() {
  const PTS = 50;

  const data1 = datasets["Simple"](PTS) as Graph;
  const data2 = datasets["Diag"](PTS) as Graph;
  const data3 = datasets["Split"](PTS) as Graph;
  const data4 = datasets["Xor"](PTS) as Graph;
  const data5 = datasets["Circle"](PTS) as Graph;
  const data6 = datasets["Spiral"](PTS) as Graph;

  console.log("=== Simple [4] ===");
  new ScalarTrain(4).train(data1, 0.5);

  console.log("\n=== Diag [4] ===");
  new ScalarTrain(4).train(data2, 0.5);

  console.log("\n=== Split [8] ===");
  new ScalarTrain(8).train(data3, 0.5);

  console.log("\n=== Xor [8] ===");
  new ScalarTrain(8).train(data4, 0.5);

  console.log("\n=== Circle [8, 8] ===");
  new ScalarTrain(8).train(data5, 0.5, 1000);

  console.log("\n=== Spiral [8] ===");
  new ScalarTrain(8).train(data6, 0.5, 1000);
}
