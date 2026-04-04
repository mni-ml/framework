import { describe, test, expect } from '@jest/globals';
import { Module, Parameter, BaseParameter } from './module.js';
import { Tensor } from './tensor.js';
import { Scalar } from './scalar.js';
import { SGD } from './optimizer.js';
import { Linear, ReLU, Sigmoid, Tanh, mseLoss, crossEntropyLoss, softmax } from './nn.js';

// ============================================================
// Helper: a minimal Module subclass for testing
// ============================================================

class LeafModule extends Module {
    p1!: Parameter<Tensor>;

    constructor() {
        super();
        this.p1 = new Parameter(Tensor.tensor([1, 2, 3]));
    }

    forward(x: Tensor): Tensor {
        return x.add(this.p1.value);
    }
}

class ParentModule extends Module {
    child1!: LeafModule;
    child2!: LeafModule;
    p_own!: Parameter<Tensor>;

    constructor() {
        super();
        this.child1 = new LeafModule();
        this.child2 = new LeafModule();
        this.p_own = new Parameter(Tensor.tensor([10]));
    }
}

class GrandparentModule extends Module {
    mid!: ParentModule;
    leaf!: LeafModule;

    constructor() {
        super();
        this.mid = new ParentModule();
        this.leaf = new LeafModule();
    }
}

// ============================================================
// Module: Proxy-based registration
// ============================================================

describe('Module proxy registration', () => {
    test('Parameter assigned to property is tracked in _parameters', () => {
        const m = new LeafModule();
        const params = m.parameters();
        expect(params).toHaveLength(1);
        expect(params[0]).toBe(m.p1);
    });

    test('Module assigned to property is tracked in _modules', () => {
        const p = new ParentModule();
        const kids = p.children();
        expect(kids).toHaveLength(2);
        expect(kids).toContain(p.child1);
        expect(kids).toContain(p.child2);
    });

    test('plain values are not tracked as parameters or modules', () => {
        class M extends Module {
            x: number = 0;
            constructor() { super(); this.x = 42; }
        }
        const m = new M();
        expect(m.parameters()).toHaveLength(0);
        expect(m.children()).toHaveLength(0);
        expect(m.x).toBe(42);
    });
});

// ============================================================
// Module: parameters() — recursive collection
// ============================================================

describe('Module.parameters()', () => {
    test('collects own parameters', () => {
        const m = new LeafModule();
        expect(m.parameters()).toHaveLength(1);
    });

    test('recursively collects child parameters', () => {
        const p = new ParentModule();
        // p_own (1) + child1.p1 (1) + child2.p1 (1) = 3
        expect(p.parameters()).toHaveLength(3);
    });

    test('recursively collects through deep hierarchy', () => {
        const g = new GrandparentModule();
        // g has no own params
        // g.mid has: p_own (1) + child1.p1 (1) + child2.p1 (1) = 3
        // g.leaf has: p1 (1) = 1
        // total = 4
        expect(g.parameters()).toHaveLength(4);
    });
});

// ============================================================
// Module: namedParameters()
// ============================================================

describe('Module.namedParameters()', () => {
    test('own parameters have simple names', () => {
        const m = new LeafModule();
        const named = m.namedParameters();
        expect(named).toHaveLength(1);
        expect(named[0]![0]).toBe('p1');
    });

    test('child parameters have dot-path names', () => {
        const p = new ParentModule();
        const named = p.namedParameters();
        const names = named.map(([n]) => n);
        expect(names).toContain('p_own');
        expect(names).toContain('child1.p1');
        expect(names).toContain('child2.p1');
    });

    test('deep hierarchy has multi-level dot paths', () => {
        const g = new GrandparentModule();
        const named = g.namedParameters();
        const names = named.map(([n]) => n);
        expect(names).toContain('mid.p_own');
        expect(names).toContain('mid.child1.p1');
        expect(names).toContain('mid.child2.p1');
        expect(names).toContain('leaf.p1');
    });
});

// ============================================================
// Module: modules() and children()
// ============================================================

describe('Module.modules() and children()', () => {
    test('modules() includes self', () => {
        const m = new LeafModule();
        const mods = m.modules();
        expect(mods).toContain(m);
    });

    test('children() does NOT include self', () => {
        const m = new LeafModule();
        expect(m.children()).not.toContain(m);
    });

    test('modules() includes all descendants', () => {
        const g = new GrandparentModule();
        const mods = g.modules();
        // g, g.mid, g.mid.child1, g.mid.child2, g.leaf = 5
        expect(mods).toHaveLength(5);
        expect(mods).toContain(g);
        expect(mods).toContain(g.mid);
        expect(mods).toContain(g.mid.child1);
        expect(mods).toContain(g.mid.child2);
        expect(mods).toContain(g.leaf);
    });

    test('children() returns only direct children', () => {
        const g = new GrandparentModule();
        const kids = g.children();
        expect(kids).toHaveLength(2);
        expect(kids).toContain(g.mid);
        expect(kids).toContain(g.leaf);
        expect(kids).not.toContain(g.mid.child1);
    });
});

// ============================================================
// Module: train() and eval()
// ============================================================

describe('Module.train() and eval()', () => {
    test('all modules start in training mode', () => {
        const g = new GrandparentModule();
        for (const m of g.modules()) {
            expect(m.training).toBe(true);
        }
    });

    test('eval() sets all modules to eval mode', () => {
        const g = new GrandparentModule();
        g.eval();
        for (const m of g.modules()) {
            expect(m.training).toBe(false);
        }
    });

    test('train() sets all modules back to training mode', () => {
        const g = new GrandparentModule();
        g.eval();
        g.train();
        for (const m of g.modules()) {
            expect(m.training).toBe(true);
        }
    });

    test('eval() does not cause infinite recursion', () => {
        const g = new GrandparentModule();
        // If eval() used this.modules() instead of this.children(),
        // this would stack overflow.
        expect(() => g.eval()).not.toThrow();
        expect(g.training).toBe(false);
    });

    test('eval on leaf module with no children', () => {
        const m = new LeafModule();
        expect(() => m.eval()).not.toThrow();
        expect(m.training).toBe(false);
    });

    test('eval on deep hierarchy — every level switches', () => {
        const g = new GrandparentModule();
        g.eval();
        expect(g.training).toBe(false);
        expect(g.mid.training).toBe(false);
        expect(g.mid.child1.training).toBe(false);
        expect(g.mid.child2.training).toBe(false);
        expect(g.leaf.training).toBe(false);
    });

    test('train/eval cycle is idempotent', () => {
        const g = new GrandparentModule();
        g.eval();
        g.eval();
        for (const m of g.modules()) {
            expect(m.training).toBe(false);
        }
        g.train();
        g.train();
        for (const m of g.modules()) {
            expect(m.training).toBe(true);
        }
    });

    test('calling eval on child does not affect parent', () => {
        const g = new GrandparentModule();
        g.mid.eval();
        expect(g.training).toBe(true);
        expect(g.leaf.training).toBe(true);
        expect(g.mid.training).toBe(false);
        expect(g.mid.child1.training).toBe(false);
        expect(g.mid.child2.training).toBe(false);
    });
});

// ============================================================
// Parameter
// ============================================================

describe('Parameter', () => {
    test('stores tensor value', () => {
        const t = Tensor.tensor([1, 2, 3]);
        const p = new Parameter(t);
        expect(p.value).toBe(t);
    });

    test('stores scalar value', () => {
        const s = new Scalar(5.0);
        const p = new Parameter(s);
        expect(p.value).toBe(s);
    });

    test('grad returns tensor grad', () => {
        const t = Tensor.tensor([1, 2, 3]);
        const p = new Parameter(t);
        expect(p.grad).toBeNull();
    });

    test('update replaces value', () => {
        const p = new Parameter(Tensor.tensor([1]));
        const newT = Tensor.tensor([2]);
        p.update(newT);
        expect(p.value).toBe(newT);
    });

    test('optional name is stored', () => {
        const p = new Parameter(Tensor.tensor([1]), 'my_weight');
        expect(p.name).toBe('my_weight');
    });
});

// ============================================================
// Linear layer
// ============================================================

describe('Linear', () => {
    test('creates weight and bias with correct shapes', () => {
        const linear = new Linear(4, 3);
        expect(linear.weight.value.shape).toEqual([4, 3]);
        expect(linear.bias.value.shape).toEqual([3]);
    });

    test('weight and bias are registered as parameters', () => {
        const linear = new Linear(4, 3);
        const params = linear.parameters();
        expect(params).toHaveLength(2);
    });

    test('namedParameters has weight and bias', () => {
        const linear = new Linear(4, 3);
        const names = linear.namedParameters().map(([n]) => n);
        expect(names).toContain('weight');
        expect(names).toContain('bias');
    });

    test('forward produces correct output shape', () => {
        const linear = new Linear(4, 3);
        const input = Tensor.rand([2, 4]); // batch=2, features=4
        const output = linear.forward(input);
        expect(output.shape).toEqual([2, 3]);
    });

    test('forward with single sample', () => {
        const linear = new Linear(5, 2);
        const input = Tensor.rand([1, 5]);
        const output = linear.forward(input);
        expect(output.shape).toEqual([1, 2]);
    });

    test('weight initialization is bounded by Xavier range', () => {
        const inF = 16;
        const linear = new Linear(inF, 8);
        const bound = 1 / Math.sqrt(inF);
        const w = linear.weight.value;
        for (let i = 0; i < w.size; i++) {
            const idx: number[] = [];
            let rem = i;
            for (let d = w.dims - 1; d >= 0; d--) {
                idx.unshift(rem % w.shape[d]!);
                rem = Math.floor(rem / w.shape[d]!);
            }
            const val = w.get(idx);
            expect(val).toBeGreaterThanOrEqual(-bound);
            expect(val).toBeLessThanOrEqual(bound);
        }
    });

    test('forward is differentiable — backward produces gradients', () => {
        const linear = new Linear(3, 2);
        const input = Tensor.rand([1, 3]);
        const output = linear.forward(input);
        const loss = output.sum();
        loss.backward();
        expect(linear.weight.value.grad).not.toBeNull();
        expect(linear.bias.value.grad).not.toBeNull();
    });

    test('train/eval works on Linear', () => {
        const linear = new Linear(3, 2);
        expect(linear.training).toBe(true);
        linear.eval();
        expect(linear.training).toBe(false);
        linear.train();
        expect(linear.training).toBe(true);
    });
});

// ============================================================
// ReLU & Sigmoid modules
// ============================================================

describe('ReLU module', () => {
    test('applies relu element-wise', () => {
        const relu = new ReLU();
        const input = Tensor.tensor([-2, -1, 0, 1, 2]);
        const output = relu.forward(input);
        expect(output.get([0])).toBe(0);
        expect(output.get([1])).toBe(0);
        expect(output.get([2])).toBe(0);
        expect(output.get([3])).toBe(1);
        expect(output.get([4])).toBe(2);
    });

    test('has no parameters', () => {
        const relu = new ReLU();
        expect(relu.parameters()).toHaveLength(0);
    });
});

describe('Sigmoid module', () => {
    test('applies sigmoid element-wise', () => {
        const sig = new Sigmoid();
        const output = sig.forward(Tensor.tensor([0]));
        expect(output.get([0])).toBeCloseTo(0.5, 5);
    });

    test('has no parameters', () => {
        const sig = new Sigmoid();
        expect(sig.parameters()).toHaveLength(0);
    });
});

describe('Tanh module', () => {
    test('tanh(0) = 0', () => {
        const tanh = new Tanh();
        const output = tanh.forward(Tensor.tensor([0]));
        expect(output.get([0])).toBeCloseTo(0, 5);
    });

    test('output is bounded in [-1, 1]', () => {
        const tanh = new Tanh();
        const output = tanh.forward(Tensor.tensor([-100, -1, 0, 1, 100]));
        for (let i = 0; i < 5; i++) {
            expect(output.get([i])).toBeGreaterThanOrEqual(-1);
            expect(output.get([i])).toBeLessThanOrEqual(1);
        }
    });

    test('matches reference values', () => {
        const tanh = new Tanh();
        const output = tanh.forward(Tensor.tensor([-2, -1, 0, 1, 2]));
        expect(output.get([0])).toBeCloseTo(Math.tanh(-2), 5);
        expect(output.get([1])).toBeCloseTo(Math.tanh(-1), 5);
        expect(output.get([2])).toBeCloseTo(Math.tanh(0), 5);
        expect(output.get([3])).toBeCloseTo(Math.tanh(1), 5);
        expect(output.get([4])).toBeCloseTo(Math.tanh(2), 5);
    });

    test('is an odd function: tanh(-x) = -tanh(x)', () => {
        const tanh = new Tanh();
        const x = Tensor.rand([10]);
        const pos = tanh.forward(x);
        const neg = tanh.forward(x.neg());
        for (let i = 0; i < 10; i++) {
            expect(neg.get([i])).toBeCloseTo(-pos.get([i]), 5);
        }
    });

    test('has no parameters', () => {
        const tanh = new Tanh();
        expect(tanh.parameters()).toHaveLength(0);
    });

    test('gradient check', () => {
        const tanh = new Tanh();
        const input = Tensor.tensor([0.5, -0.3, 1.2, -0.8]);
        const output = tanh.forward(input);
        output.sum().backward();

        const eps = 1e-5;
        for (let i = 0; i < 4; i++) {
            const orig = input.get([i]);
            input.set([i], orig + eps);
            const plus = tanh.forward(input).sum().item();
            input.set([i], orig - eps);
            const minus = tanh.forward(input).sum().item();
            input.set([i], orig);

            const numerical = (plus - minus) / (2 * eps);
            const analytical = input.grad!.get([i]);
            expect(analytical).toBeCloseTo(numerical, 4);
        }
    });
});

// ============================================================
// Loss functions
// ============================================================

describe('mseLoss', () => {
    test('loss is zero when input equals target', () => {
        const t = Tensor.tensor([1, 2, 3]);
        const loss = mseLoss(t, t);
        expect(loss.item()).toBeCloseTo(0, 5);
    });

    test('correct value for known input', () => {
        const input = Tensor.tensor([1, 2, 3]);
        const target = Tensor.tensor([1, 1, 1]);
        // diffs: [0, 1, 2], squares: [0, 1, 4], mean = 5/3
        const loss = mseLoss(input, target);
        expect(loss.item()).toBeCloseTo(5 / 3, 5);
    });

    test('loss is differentiable', () => {
        const input = Tensor.rand([4]);
        const target = Tensor.rand([4]);
        const loss = mseLoss(input, target);
        loss.backward();
        expect(input.grad).not.toBeNull();
    });

    test('gradient check', () => {
        const input = Tensor.rand([3]);
        const target = Tensor.tensor([0.5, 0.5, 0.5]);
        const loss = mseLoss(input, target);
        loss.backward();

        const eps = 1e-5;
        for (let i = 0; i < 3; i++) {
            const orig = input.get([i]);
            input.set([i], orig + eps);
            const plus = mseLoss(input, target).item();
            input.set([i], orig - eps);
            const minus = mseLoss(input, target).item();
            input.set([i], orig);

            const numerical = (plus - minus) / (2 * eps);
            const analytical = input.grad!.get([i]);
            expect(analytical).toBeCloseTo(numerical, 3);
        }
    });
});

describe('crossEntropyLoss', () => {
    test('loss is lower for correct prediction', () => {
        const correct = Tensor.tensor([[10, 0, 0]]);   // high logit on class 0
        const wrong = Tensor.tensor([[0, 0, 10]]);      // high logit on class 2
        const target = Tensor.tensor([[1, 0, 0]]);      // one-hot for class 0

        const lossCorrect = crossEntropyLoss(correct, target).item();
        const lossWrong = crossEntropyLoss(wrong, target).item();
        expect(lossCorrect).toBeLessThan(lossWrong);
    });

    test('loss is non-negative', () => {
        const input = Tensor.rand([1, 5]);
        const target = Tensor.tensor([[1, 0, 0, 0, 0]]);
        const loss = crossEntropyLoss(input, target).item();
        expect(loss).toBeGreaterThanOrEqual(-1e-7);
    });

    test('loss is differentiable', () => {
        const input = Tensor.rand([2, 3]);
        const target = Tensor.tensor([[1, 0, 0], [0, 1, 0]]);
        const loss = crossEntropyLoss(input, target);
        loss.backward();
        expect(input.grad).not.toBeNull();
        expect(input.grad!.shape).toEqual([2, 3]);
    });
});

// ============================================================
// Composite model: multi-layer network with train/eval
// ============================================================

describe('composite model', () => {
    class MLP extends Module {
        layer1!: Linear;
        relu!: ReLU;
        layer2!: Linear;

        constructor(inputDim: number, hiddenDim: number, outputDim: number) {
            super();
            this.layer1 = new Linear(inputDim, hiddenDim);
            this.relu = new ReLU();
            this.layer2 = new Linear(hiddenDim, outputDim);
        }

        forward(x: Tensor): Tensor {
            let out = this.layer1.forward(x);
            out = this.relu.forward(out);
            return this.layer2.forward(out);
        }
    }

    test('parameters from all layers are collected', () => {
        const model = new MLP(4, 3, 2);
        // layer1: weight + bias = 2, layer2: weight + bias = 2, relu: 0
        expect(model.parameters()).toHaveLength(4);
    });

    test('namedParameters has correct dot-path names', () => {
        const model = new MLP(4, 3, 2);
        const names = model.namedParameters().map(([n]) => n);
        expect(names).toContain('layer1.weight');
        expect(names).toContain('layer1.bias');
        expect(names).toContain('layer2.weight');
        expect(names).toContain('layer2.bias');
    });

    test('modules() includes all sub-modules', () => {
        const model = new MLP(4, 3, 2);
        const mods = model.modules();
        // model, layer1, relu, layer2 = 4
        expect(mods).toHaveLength(4);
        expect(mods).toContain(model);
        expect(mods).toContain(model.layer1);
        expect(mods).toContain(model.relu);
        expect(mods).toContain(model.layer2);
    });

    test('eval() propagates to all sub-modules', () => {
        const model = new MLP(4, 3, 2);
        model.eval();
        expect(model.training).toBe(false);
        expect(model.layer1.training).toBe(false);
        expect(model.relu.training).toBe(false);
        expect(model.layer2.training).toBe(false);
    });

    test('train() propagates to all sub-modules', () => {
        const model = new MLP(4, 3, 2);
        model.eval();
        model.train();
        expect(model.training).toBe(true);
        expect(model.layer1.training).toBe(true);
        expect(model.relu.training).toBe(true);
        expect(model.layer2.training).toBe(true);
    });

    test('forward produces correct shape', () => {
        const model = new MLP(4, 3, 2);
        const input = Tensor.rand([5, 4]);
        const output = model.forward(input);
        expect(output.shape).toEqual([5, 2]);
    });

    test('backward produces gradients on all parameters', () => {
        const model = new MLP(4, 3, 2);
        const input = Tensor.rand([2, 4]);
        const output = model.forward(input);
        const loss = output.sum();
        loss.backward();

        for (const [name, param] of model.namedParameters()) {
            expect(param.grad).not.toBeNull();
        }
    });
});

// ============================================================
// End-to-end: SGD training loop
// ============================================================

describe('end-to-end training', () => {
    test('MSE loss decreases over SGD steps', () => {
        const linear = new Linear(2, 1);
        const optimizer = new SGD(
            linear.parameters() as Parameter<Tensor>[],
            0.01
        );

        const input = Tensor.tensor([[1, 0], [0, 1], [1, 1], [0, 0]]);
        const target = Tensor.tensor([[1], [0], [1], [0]]);

        let prevLoss = Infinity;
        for (let epoch = 0; epoch < 50; epoch++) {
            optimizer.zeroGrad();
            const output = linear.forward(input);
            const loss = mseLoss(output, target);
            loss.backward();
            optimizer.step();

            if (epoch % 10 === 0) {
                const lossVal = loss.item();
                expect(lossVal).toBeLessThan(prevLoss + 1e-7);
                prevLoss = lossVal;
            }
        }

        expect(prevLoss).toBeLessThan(1.0);
    });
});
