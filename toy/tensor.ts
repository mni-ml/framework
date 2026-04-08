import type {
    Shape,
} from './tensor_data.js'

import {
    TensorData,
    shapeProduct,
} from './tensor_data.js'
import * as tensorFunctions from './tensor_functions.js'
import { tensorMap } from './tensor_ops.js'
import { 
    TensorContext, 
    TensorHistory, 
    TensorFunction,
    Neg as NegFn,
    Sigmoid as SigmoidFn,
    ReLU as ReLUFn,
    Log as LogFn,
    Exp as ExpFn,
    Inv as InvFn,
    Add as AddFn,
    Mul as MulFn,
    LT as LTFn,
    EQ as EQFn,
    Sum as SumFn,
    Permute as PermuteFn,
    View as ViewFn,
    Contiguous as ContiguousFn,
} from './tensor_functions.js';
import { backPropagateTensor } from './autodiff.js';

export type TensorLike = number | Tensor;

export class Tensor {
    private _data: TensorData;
    grad: Tensor | null = null;
    history: TensorHistory | null = null;

    constructor(data: TensorData, history: TensorHistory | null = null) {
        this._data = data;
        this.history = history;
    }

    isLeaf(): boolean {
        return !this.history?.lastFn;
    }

    requiresGrad(): boolean {
        return this.history !== null;
    }

    accumulateGrad(grad: Tensor): void {
        if (this.grad === null) {
            this.grad = grad;
        } else {
            this.grad = this.grad.add(grad);
        }
    }

    chainRule(gradOutput: Tensor): [Tensor, Tensor][] {
        const h = this.history;
        if (!h || !h.lastFn || !h.ctx) {
            throw new Error("Cannot call chainRule on leaf tensor");
        }

        const gradients: Tensor[] = h.lastFn.backward(h.ctx, gradOutput);

        if (gradients.length !== h.inputs.length) {
            throw new Error(
                `Backward returned ${gradients.length} gradients but expected ${h.inputs.length} for ${h.inputs.length} inputs`
            );
        }

        const result: [Tensor, Tensor][] = [];
        for (let i = 0; i < h.inputs.length; i++) {
            result.push([h.inputs[i]!, gradients[i]!]);
        }
        return result;
    }

    get parents(): Tensor[] {
        return this.history?.inputs ?? [];
    }

    static apply(fn: typeof TensorFunction, ...vals: TensorLike[]): Tensor {
        const tensors: Tensor[] = vals.map(v =>
            v instanceof Tensor ? v : Tensor.tensor(v)
        )

        const ctx = new TensorContext();
        const result = fn.forward(ctx, ...tensors);

        const history = new TensorHistory(fn, ctx, tensors);
        result.history = history;

        return result;
    }

    backward(gradOutput?: Tensor): void {
        if (gradOutput == undefined) {
            gradOutput = Tensor.ones(this.shape);
        }
        backPropagateTensor(this, gradOutput);
    }

    static tensor(values: any, shape?: Shape): Tensor {
        if (typeof values === 'number') {
            const storage = new Float64Array([values]);
            return new Tensor(new TensorData(storage, []));
        }

        const { flat, inferredShape } = flattenArray(values);
        const finalShape = shape ?? inferredShape;

        if (shapeProduct(finalShape) != flat.length) {
            throw new Error(
                `Shape is incompatible with flat array size`
            );
        }

        const storage = new Float64Array(flat);
        return new Tensor(new TensorData(storage, finalShape));
    }

    static zeros(shape: Shape) : Tensor {
        return new Tensor(TensorData.zeros(shape));
    }

    static ones(shape: Shape): Tensor {
        const size = shapeProduct(shape);
        const storage = new Float64Array(size).fill(1);
        return new Tensor(new TensorData(storage, shape));
    }

    static rand(shape: Shape): Tensor {
        const size = shapeProduct(shape);
        const storage = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            storage[i] = Math.random();
        }
        return new Tensor(new TensorData(storage, shape));
    }

    get size(): number {
        return this._data.size;
    }

    get dims(): number {
        return this._data.dims;
    }

    get shape(): Shape {
        return this._data.shape;
    }

    get data(): TensorData {
        return this._data;
    }

    private static _ensureTensor(value: number | Tensor) : Tensor {
        if (value instanceof Tensor) {
            return value;
        }

        return Tensor.tensor(value);
    }

    neg(): Tensor {
        return Tensor.apply(NegFn, this);
    }

    sigmoid(): Tensor {
        return Tensor.apply(SigmoidFn, this);
    }

    relu(): Tensor {
        return Tensor.apply(ReLUFn, this);
    }

    log(): Tensor {
        return Tensor.apply(LogFn, this);
    }

    exp(): Tensor {
        return Tensor.apply(ExpFn, this);
    }

    inv(): Tensor {
        return Tensor.apply(InvFn, this);
    }

    add(other: number | Tensor): Tensor {
        return Tensor.apply(AddFn, this, Tensor._ensureTensor(other));
    }

    sub(other: number | Tensor): Tensor {
        // a - b = a + (-b)
        return this.add(Tensor._ensureTensor(other).neg());
    }

    mul(other: number | Tensor): Tensor {
        return Tensor.apply(MulFn, this, Tensor._ensureTensor(other));
    }

    lt(other: number | Tensor): Tensor {
        return Tensor.apply(LTFn, this, Tensor._ensureTensor(other));
    }

    eq(other: number | Tensor): Tensor {
        return Tensor.apply(EQFn, this, Tensor._ensureTensor(other));
    }

    gt(other: number | Tensor): Tensor {
        // a > b is equivalent to b < a
        const b = Tensor._ensureTensor(other);
        return Tensor.apply(LTFn, b, this);
    }

    is_close(other: number | Tensor): Tensor {
        // Note: is_close is a comparison operation and does not support gradients
        const b = Tensor._ensureTensor(other);
        return new Tensor(tensorFunctions.isClose(this._data, b._data));
    }

    radd(other: number | Tensor): Tensor {
        return this.add(other);
    }

    rmul(other: number | Tensor): Tensor {
        return this.mul(other);
    }

    sum(dim?: number): Tensor {
        if (dim === undefined) {
            if (this.dims === 0) {
                return this;
            }
            let result: Tensor = this;
            for (let d = result.dims - 1; d >= 0; d--) {
                result = Tensor.apply(SumFn(d), result);
            }
            return result.view();
        }

        if (dim < 0 || dim >= this.dims) {
            throw new Error(`Invalid dimension ${dim} for tensor with ${this.dims} dimensions`);
        }

        return Tensor.apply(SumFn(dim), this);
    }

    mean(dim?: number): Tensor {
        if (dim === undefined) {
            const s = this.sum();
            const count = this.size;
            return s.mul(1 / count);
        }

        if (dim < 0 || dim >= this.dims) {
            throw new Error(`Invalid dimension ${dim} for tensor with ${this.dims} dimensions`);
        }

        const s = this.sum(dim);
        const count = this._data.shape[dim]!;
        return s.mul(1 / count);
    }

    // Note: all() is a boolean reduction and does not support gradients
    all(dim?: number): Tensor {
        if (dim === undefined) {
            let result = this._data;
            for (let d = 0; d < result.dims; d++) {
                result = tensorFunctions.prod(result, d);
            }
            const val = result.storage[0]! !== 0 ? 1 : 0;
            return Tensor.tensor(val);
        }

        if (dim < 0 || dim >= this.dims) {
            throw new Error(`Invalid dimension ${dim} for tensor with ${this.dims} dimensions`);
        }

        const p = tensorFunctions.prod(this._data, dim);
        const toBoolean = (x: number) => (x !== 0 ? 1 : 0);
        const out = TensorData.zeros(p.shape);
        const mapFn = tensorMap(toBoolean);
        mapFn(out.storage, out.shape, out.strides, p.storage, p.shape, p.strides);
        return new Tensor(out);
    }

    permute(...order: number[]): Tensor {
        return Tensor.apply(PermuteFn(order), this);
    }

    view(...shape: number[]): Tensor {
        return Tensor.apply(ViewFn(shape), this);
    }

    contiguous(): Tensor {
        return Tensor.apply(ContiguousFn, this);
    }

    zero_grad_():void {
        this.grad = null;
    }

    get(idx: number[]): number {
        return this._data.get(idx);
    }

    set(idx: number[], value: number): void {
        this._data.set(idx, value);
    }

    item(): number {
        if (this.size !== 1) {
            throw new Error('item() only works for tensors with exactly one element');
        }
        
        const idx = new Array(this.dims).fill(0);
        return this._data.get(idx);
    }

    toArray(): any {
        return buildNestedArray(this._data, 0, new Array(this.dims).fill(0));
    }

    toString(): string {
        if (this.dims === 0) {
            return `Tensor(${this._data.storage[0]})`;
        }
        return `Tensor(${JSON.stringify(this.toArray())}, shape=[${this.shape.join(', ')}])`;
    }
}

function flattenArray(arr: any): { flat: number[]; inferredShape: number[] } {
    const shape: number[] = [];
    let current: any = arr;
    while (Array.isArray(current)) {
        shape.push(current.length);
        current = current[0];
    }

    const flat: number[] = [];
    flattenRecursive(arr, flat);

    return {flat, inferredShape: shape};
}

function flattenRecursive(arr: any, out: number[]): void {
    if (Array.isArray(arr)) {
        for (const item of arr) {
            flattenRecursive(item, out);
        }
    } else {
        out.push(arr);
    }
}

function buildNestedArray(data: TensorData, dim: number, idx: number[]): any {
    if (dim == data.dims) {
        return data.get(idx);
    }

    const result: any[] = [];
    for (let i = 0; i < data.shape[dim]!; i++) {
        idx[dim] = i;
        result.push(buildNestedArray(data, dim + 1, idx));
    }
    return result;
}