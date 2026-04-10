import {
    Tensor,
    Module, Parameter,
    Linear, ReLU,
    Adam, SGD, GradScaler,
    mseLoss,
} from '../dist/index.js';
import { assert, assertClose, section } from './helpers.js';

// ============================================================
// Module system
// ============================================================

section('Module system');

class TestNet extends Module {
    l1: any;
    l2: any;
    relu: any;
    constructor() {
        super();
        this.l1 = new Linear(3, 4);
        this.l2 = new Linear(4, 2);
        this.relu = new ReLU();
    }
    forward(x: any) {
        return this.l2.forward(this.relu.forward(this.l1.forward(x)));
    }
}

const net = new TestNet();

const params = net.parameters();
assert(params.length === 4, 'TestNet has 4 parameters (2 weights + 2 biases)');

const named = net.namedParameters();
assert(named.length === 4, 'namedParameters count');
const names = named.map(([n]: [string, any]) => n);
assert(names.some((n: string) => n.includes('l1')), 'namedParameters includes l1');
assert(names.some((n: string) => n.includes('l2')), 'namedParameters includes l2');

const kids = net.children();
assert(kids.length === 3, 'TestNet has 3 children (l1, l2, relu)');

const allMods = net.modules();
assert(allMods.length >= 4, 'modules() includes self + children');

net.eval();
assert(net.training === false, 'eval sets training=false');
net.train();
assert(net.training === true, 'train sets training=true');

const netInput = Tensor.rand([2, 3]);
const netOut = net.forward(netInput);
assert(netOut.shape[0] === 2 && netOut.shape[1] === 2, 'TestNet output shape');

// ============================================================
// Optimizers
// ============================================================

section('Optimizers');

// SGD
const sgdParam = Tensor.fromFloat32(new Float32Array([5, 5, 5, 5]), [2, 2]).setRequiresGrad(true);
const sgdParamObj = new Parameter(sgdParam);
const sgdTarget = Tensor.zeros([2, 2]);
const sgdLoss = sgdParamObj.value.sub(sgdTarget).pow(2).mean();
sgdLoss.backward();
const sgd = new SGD([sgdParamObj], 0.1);
const sgdBefore = sgdParamObj.value.toFloat32()[0];
sgd.step();
const sgdAfter = sgdParamObj.value.toFloat32()[0];
assert(sgdAfter < sgdBefore, 'SGD step reduces parameter toward target');
sgd.zeroGrad();

// Adam
const adamParam = Tensor.fromFloat32(new Float32Array([5, 5, 5, 5]), [2, 2]).setRequiresGrad(true);
const adamParamObj = new Parameter(adamParam);
const adamTarget = Tensor.zeros([2, 2]);
const adamLoss = adamParamObj.value.sub(adamTarget).pow(2).mean();
adamLoss.backward();
const adam = new Adam([adamParamObj], { lr: 0.01 });
adam.step();
const adamAfter = adamParamObj.value.toFloat32()[0];
assert(adamAfter !== 5, 'Adam step changes parameter');
adam.zeroGrad();

// Adam returns grad norm
const adamParam2 = Tensor.fromFloat32(new Float32Array([3, 3]), [2]).setRequiresGrad(true);
const adamParamObj2 = new Parameter(adamParam2);
const adamLoss2 = adamParamObj2.value.pow(2).sum();
adamLoss2.backward();
const adam2 = new Adam([adamParamObj2], { lr: 0.01 });
const gradNorm = adam2.step();
assert(typeof gradNorm === 'number', 'Adam.step() returns grad norm');

// GradScaler
const scaler = new GradScaler({ initScale: 1024 });
assert(scaler.getScale() === 1024, 'GradScaler initial scale');

const gsLossInput = Tensor.fromFloat32(new Float32Array([2, 3]), [2]);
const scaledLoss = scaler.scaleLoss(gsLossInput);
const scaledData = scaledLoss.toFloat32();
assertClose(scaledData[0], 2 * 1024, 1e-1, 'scaleLoss scales by initScale');
assertClose(scaledData[1], 3 * 1024, 1e-1, 'scaleLoss scales second element');

// ============================================================
// End-to-end training loop
// ============================================================

section('End-to-end training');

const trainX = Tensor.fromFloat32(new Float32Array([0, 1, 2, 3, 4, 5]), [6, 1]);
const trainY = Tensor.fromFloat32(new Float32Array([1, 3, 5, 7, 9, 11]), [6, 1]);
const regNet = new Linear(1, 1);
const regOptim = new Adam(regNet.parameters(), { lr: 0.05 });

let earlyLoss: number | null = null;
for (let i = 0; i < 200; i++) {
    regOptim.zeroGrad();
    const pred = regNet.forward(trainX);
    const loss = mseLoss(pred, trainY);
    if (i === 10) earlyLoss = loss.toFloat32()[0];
    loss.backward();
    regOptim.step();
}
const finalPred = regNet.forward(trainX);
const finalLoss = mseLoss(finalPred, trainY).toFloat32()[0];
assert(finalLoss < earlyLoss!, 'training reduces loss');
assert(finalLoss < 1.0, 'training converges to low loss');

