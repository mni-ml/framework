import type {
    Shape
} from './tensor_data.js'

import {
    TensorData,
    shapeProduct,
    shapeBroadcast,
    strides,
} from './tensor_data.js';
import { fastTensorMap as tensorMap, fastTensorZip as tensorZip, fastTensorReduce as tensorReduce } from './fast_ops.js';
import * as operators from './operators.js'
import { Tensor } from './tensor.js';



function zeros(shape: Shape): TensorData {
    return TensorData.zeros(shape);
}

export function neg(a: TensorData): TensorData {
    const out = zeros(a.shape);
    const mapFn = tensorMap(operators.neg);
    mapFn(out.storage, out.shape, out.strides, a.storage, a.shape, a.strides);
    return out;
}

export function sigmoid(a: TensorData): TensorData {
    const out = zeros(a.shape);
    const mapFn = tensorMap(operators.sigmoid);
    mapFn(out.storage, out.shape, out.strides, a.storage, a.shape, a.strides);
    return out;
}

export function relu(a: TensorData): TensorData {
    const out = zeros(a.shape);
    const mapFn = tensorMap(operators.relu);
    mapFn(out.storage, out.shape, out.strides, a.storage, a.shape, a.strides);
    return out;
}

export function log(a: TensorData): TensorData {
    const out = zeros(a.shape);
    const mapFn = tensorMap(operators.log);
    mapFn(out.storage, out.shape, out.strides, a.storage, a.shape, a.strides);
    return out;
}

export function exp(a: TensorData): TensorData {
    const out = zeros(a.shape);
    const mapFn = tensorMap(operators.exp);
    mapFn(out.storage, out.shape, out.strides, a.storage, a.shape, a.strides);
    return out;
}

export function id(a: TensorData): TensorData {
    const out = zeros(a.shape);
    const mapFn = tensorMap(operators.id);
    mapFn(out.storage, out.shape, out.strides, a.storage, a.shape, a.strides);
    return out;
}

export function inv(a: TensorData): TensorData {
    const out = zeros(a.shape);
    const mapFn = tensorMap(operators.inv);
    mapFn(out.storage, out.shape, out.strides, a.storage, a.shape, a.strides);
    return out;
}

export function add(a: TensorData, b: TensorData): TensorData {
    const outShape = shapeBroadcast(a.shape, b.shape);
    const out = zeros(outShape);
    const zipFn = tensorZip(operators.add);
    zipFn(
        out.storage, out.shape, out.strides,
        a.storage, a.shape, a.strides,
        b.storage, b.shape, b.strides
    );
    return out;
}

export function mul(a: TensorData, b: TensorData): TensorData {
    const outShape = shapeBroadcast(a.shape, b.shape);
    const out = zeros(outShape);
    const zipFn = tensorZip(operators.mul);
    zipFn(
        out.storage, out.shape, out.strides,
        a.storage, a.shape, a.strides,
        b.storage, b.shape, b.strides
    );
    return out;
}

export function lt(a: TensorData, b: TensorData): TensorData {
    const outShape = shapeBroadcast(a.shape, b.shape);
    const out = zeros(outShape);
    const zipFn = tensorZip(operators.lt);
    zipFn(
        out.storage, out.shape, out.strides,
        a.storage, a.shape, a.strides,
        b.storage, b.shape, b.strides
    );
    return out;
}

export function eq(a: TensorData, b: TensorData): TensorData {
    const outShape = shapeBroadcast(a.shape, b.shape);
    const out = zeros(outShape);
    const zipFn = tensorZip(operators.eq);
    zipFn(
        out.storage, out.shape, out.strides,
        a.storage, a.shape, a.strides,
        b.storage, b.shape, b.strides
    );
    return out;
}

export function isClose(a: TensorData, b: TensorData): TensorData {
    const outShape = shapeBroadcast(a.shape, b.shape);
    const out = zeros(outShape);
    const zipFn = tensorZip(operators.isClose);
    zipFn(
        out.storage, out.shape, out.strides,
        a.storage, a.shape, a.strides,
        b.storage, b.shape, b.strides
    );
    return out;
}

export function sum(a: TensorData, dim: number): TensorData {
    const outShape = a.shape.map((s, i) => (i === dim ? 1 : s));
    const out = zeros(outShape);
    const reduceFn = tensorReduce(operators.add);
    reduceFn(
        out.storage, out.shape, out.strides,
        a.storage, a.shape, a.strides,
        dim
    );
    return out;
}

export function prod(a: TensorData, dim: number): TensorData {
    const outShape = a.shape.map((s, i) => (i === dim ? 1 : s));
    const out = zeros(outShape);
    const reduceFn = tensorReduce(operators.mul);
    reduceFn(
        out.storage, out.shape, out.strides,
        a.storage, a.shape, a.strides,
        dim
    );
    return out;
}

export function permute(a: TensorData, order: number[]): TensorData {
    return a.permute(...order);
}

export function view(a: TensorData, shape: Shape): TensorData {
    const expectedStrides = strides(a.shape);
    const isContiguous = a.strides.every((s, i) => s === expectedStrides[i]);
    
    if (!isContiguous) {
        throw new Error (
            'Cannot view a non-contiguous tensor'
        );
    }

    const newSize = shapeProduct(shape);
    if (newSize !== a.size) {
        throw new Error(
            `Size mismatch of tensor and shape [${shape.join(', ')}]`
        );
    }

    return new TensorData(a.storage, shape);
}

export function contiguous(a: TensorData): TensorData {
    const expectedStrides = strides(a.shape);
    const isContiguous = a.strides.every((s, i) => s === expectedStrides[i]);

    if (isContiguous) {
        return a;
    }

    return id(a);
}

export class TensorContext {
    private _savedTensors: Tensor[] = [];

    saveForBackward(...tensors: Tensor[]): void {
        this._savedTensors = tensors;
    }

    get savedTensors(): Tensor[] {
        return this._savedTensors;
    }
}

export class TensorHistory {
    constructor (
        public lastFn: typeof TensorFunction | null = null,
        public ctx: TensorContext | null = null,
        public inputs: Tensor[] = []
    ) {}
}

export abstract class TensorFunction {
    static forward(ctx: TensorContext, ...inputs: Tensor[]): Tensor {
        throw new Error("forward not implemented");
    }

    static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
        throw new Error("backward not implemented");
    }
}

export class Neg extends TensorFunction {
    static forward(ctx: TensorContext, a: Tensor): Tensor {
        return new Tensor(neg(a.data));
    }
    static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
        return [gradOutput.neg()];
    }
}

export class Sigmoid extends TensorFunction {
    static forward(ctx: TensorContext, a: Tensor): Tensor {
        const result = new Tensor(sigmoid(a.data));
        ctx.saveForBackward(result);  // Save output
        return result;
    }
    static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
        const [sigResult] = ctx.savedTensors;
        // grad * sig * (1 - sig)
        const ones = Tensor.ones(sigResult!.shape);
        return [gradOutput.mul(sigResult!).mul(ones.sub(sigResult!))];
    }
}

export class ReLU extends TensorFunction {
    static forward(ctx: TensorContext, a: Tensor): Tensor {
        ctx.saveForBackward(a);  // Save input
        return new Tensor(relu(a.data));
    }
    static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
        const [a] = ctx.savedTensors;
        const mask = new Tensor(lt(Tensor.zeros(a!.shape).data, a!.data));
        return [gradOutput.mul(mask)];
    }
}

export class Log extends TensorFunction {
    static forward(ctx: TensorContext, a: Tensor): Tensor {
        ctx.saveForBackward(a);
        return new Tensor(log(a.data));
    }
    static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
        const [a] = ctx.savedTensors;
        // grad / a
        return [gradOutput.mul(a!.inv())];
    }
}

export class Exp extends TensorFunction {
    static forward(ctx: TensorContext, a: Tensor): Tensor {
        const result = new Tensor(exp(a.data));
        ctx.saveForBackward(result);  // Save output
        return result;
    }
    static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
        const [expResult] = ctx.savedTensors;
        return [gradOutput.mul(expResult!)];
    }
}

export class Inv extends TensorFunction {
    static forward(ctx: TensorContext, a: Tensor): Tensor {
        ctx.saveForBackward(a);
        return new Tensor(inv(a.data));
    }
    static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
        const [a] = ctx.savedTensors;
        // -grad / a^2
        return [gradOutput.neg().mul(a!.mul(a!).inv())];
    }
}

function unbroadcast(grad: Tensor, originalShape: Shape): Tensor {
    let result = grad;

    while (result.dims > originalShape.length) {
        result = result.sum(0);
        const newShape = result.shape.slice(1);
        result = result.view(...newShape);
    }

    for (let i = 0; i < originalShape.length; i++) {
        if (originalShape[i] === 1 && result.shape[i]! > 1) {
            result = result.sum(i);
        }
    }

    return result;
}

export class Add extends TensorFunction {
    static forward(ctx: TensorContext, a: Tensor, b: Tensor): Tensor {
        ctx.saveForBackward(a, b);
        return new Tensor(add(a.data, b.data));
    }
    static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
        const [a, b] = ctx.savedTensors;
        return [
            unbroadcast(gradOutput, a!.shape),
            unbroadcast(gradOutput, b!.shape)
        ]
    }
}

export class Mul extends TensorFunction {
    static forward(ctx: TensorContext, a: Tensor, b: Tensor): Tensor {
        ctx.saveForBackward(a, b);
        return new Tensor(mul(a.data, b.data));
    }
    static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
        const [a, b] = ctx.savedTensors;
        return [
            unbroadcast(gradOutput.mul(b!), a!.shape),
            unbroadcast(gradOutput.mul(a!), b!.shape)
        ];
    }
}

export class LT extends TensorFunction {
    static forward(ctx: TensorContext, a: Tensor, b: Tensor): Tensor {
        ctx.saveForBackward(a, b);
        return new Tensor(lt(a.data, b.data));
    }
    static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
        const [a, b] = ctx.savedTensors;
        return [Tensor.zeros(a!.shape), Tensor.zeros(b!.shape)];
    }
}

export class EQ extends TensorFunction {
    static forward(ctx: TensorContext, a: Tensor, b: Tensor): Tensor {
        ctx.saveForBackward(a, b);
        return new Tensor(eq(a.data, b.data));
    }
    static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
        const [a, b] = ctx.savedTensors;
        return [Tensor.zeros(a!.shape), Tensor.zeros(b!.shape)];
    }
}

export function Sum(dim: number): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: TensorContext, a: Tensor): Tensor {
            ctx.saveForBackward(a);
            return new Tensor(sum(a.data, dim));
        }
        static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
            const [a] = ctx.savedTensors;
            return [gradOutput.mul(Tensor.ones(a!.shape))];
        }
    };
}

export function Permute(order: number[]): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: TensorContext, a: Tensor): Tensor {
            ctx.saveForBackward(a);
            return new Tensor(permute(a.data, order));
        }
        static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
            const inverseOrder = new Array(order.length);
            for (let i = 0; i < order.length; i++) {
                inverseOrder[order[i]!] = i;
            }
            return [gradOutput.permute(...inverseOrder)];
        }
    };
}

export function View(newShape: Shape): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: TensorContext, a: Tensor): Tensor {
            ctx.saveForBackward(a);
            return new Tensor(view(a.data, newShape));
        }
        static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
            const [a] = ctx.savedTensors;
            return [gradOutput.contiguous().view(...a!.shape)];
        }
    };
}

export class Contiguous extends TensorFunction {
    static forward(ctx: TensorContext, a: Tensor): Tensor {
        return new Tensor(contiguous(a.data));
    }
    static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
        return [gradOutput];
    }
}