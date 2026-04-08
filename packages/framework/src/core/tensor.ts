import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname_f = dirname(fileURLToPath(import.meta.url));

function loadNative() {
    const require = createRequire(import.meta.url);
    const platform = process.platform;
    const arch = process.arch;
    let suffix: string;
    if (platform === 'darwin' && arch === 'arm64') {
        suffix = 'darwin-arm64';
    } else if (platform === 'linux' && arch === 'x64') {
        suffix = 'linux-x64-gnu';
    } else {
        suffix = `${platform}-${arch}`;
    }
    const ext = platform === 'darwin' ? 'dylib' : 'so';
    const candidates = [
        join(__dirname_f, '..', '..', 'native', `mni-framework-native.${suffix}.node`),
        join(__dirname_f, '..', '..', 'native', 'target', 'release', `libmni_framework_native.${ext}`),
        join(__dirname_f, '..', 'native', `mni-framework-native.${suffix}.node`),
        join(__dirname_f, '..', 'native', 'target', 'release', `libmni_framework_native.${ext}`),
    ];
    for (const p of candidates) {
        if (existsSync(p)) {
            return require(p);
        }
    }
    throw new Error(`Native addon not found for ${platform}-${arch}. Tried: ${candidates.join(', ')}`);
}

export const native: any = loadNative();

export type Shape = number[];
type TensorDataLike = {
    shape: readonly number[];
    storage: ArrayLike<number>;
};

type NestedNumberArray = number | NestedNumberArray[];

function isTensorDataLike(value: unknown): value is TensorDataLike {
    return typeof value === "object"
        && value !== null
        && "shape" in value
        && "storage" in value;
}

function inferShape(value: NestedNumberArray): Shape {
    if (!Array.isArray(value)) {
        return [];
    }

    if (value.length === 0) {
        return [0];
    }

    return [value.length, ...inferShape(value[0]!)];
}

function flattenNestedArray(value: NestedNumberArray, out: number[]): void {
    if (Array.isArray(value)) {
        for (const entry of value) {
            flattenNestedArray(entry, out);
        }
        return;
    }

    out.push(value);
}

export class Tensor {
    readonly _id: number;
    private _shape: Shape;

    constructor(id: number, shape?: Shape);
    constructor(data: TensorDataLike);
    constructor(source: number | TensorDataLike, shape?: Shape) {
        if (typeof source === "number") {
            this._id = source;
            this._shape = shape ?? native.tensorShape(source).map(Number);
            return;
        }

        const inferredShape = [...source.shape];
        const data = Float32Array.from(source.storage);
        this._id = native.fromFloat32(data, inferredShape.map(Number));
        this._shape = inferredShape;
    }

    get shape(): Shape { return this._shape; }
    get size(): number { return this._shape.reduce((a: number, b: number) => a * b, 1); }
    get dims(): number { return this._shape.length; }

    // Compatibility stubs for old code that accesses .data.storage or .history
    get data(): { storage: Float32Array } {
        return { storage: this.toFloat32() };
    }
    set history(_v: any) { /* no-op: autograd is in Rust */ }
    get history(): null { return null; }
    get parents(): Tensor[] { return []; }

    // ---- Gradient ----

    get grad(): Tensor | null {
        const gid = native.getGrad(this._id);
        if (gid == null) return null;
        return new Tensor(gid);
    }
    set grad(_v: Tensor | null) { /* managed by Rust */ }

    backward(): void {
        native.backward(this._id);
    }

    isLeaf(): boolean {
        return true;
    }

    accumulateGrad(_grad: Tensor): void {
        // no-op: gradients are tracked inside the native backend
    }

    async chainRule(_grad: Tensor): Promise<Array<[Tensor, Tensor]>> {
        return [];
    }

    requiresGrad(): boolean {
        return native.getGrad(this._id) != null;
    }

    // ---- Data transfer ----

    toFloat32(): Float32Array {
        return native.toFloat32(this._id);
    }

    item(): number {
        return native.getScalar(this._id);
    }

    get(indices: number[]): number {
        const data = this.toFloat32();
        let flat = 0;
        let stride = 1;
        for (let i = this._shape.length - 1; i >= 0; i--) {
            flat += indices[i] * stride;
            stride *= this._shape[i];
        }
        return data[flat];
    }

    free(): void {
        native.freeTensor(this._id);
    }

    // ---- Creation (static) ----

    static fromFloat32(data: Float32Array, shape: Shape): Tensor {
        const id = native.fromFloat32(data, shape.map(Number));
        return new Tensor(id, shape);
    }

    static tensor(data: NestedNumberArray[]): Tensor;
    static tensor(data: NestedNumberArray): Tensor;
    static tensor(data: NestedNumberArray): Tensor {
        const shape = inferShape(data);
        const flat: number[] = [];
        flattenNestedArray(data, flat);
        return Tensor.fromFloat32(Float32Array.from(flat), shape);
    }

    static zeros(shape: Shape): Tensor {
        const id = native.zeros(shape.map(Number));
        return new Tensor(id, shape);
    }

    static ones(shape: Shape): Tensor {
        const id = native.ones(shape.map(Number));
        return new Tensor(id, shape);
    }

    static rand(shape: Shape): Tensor {
        const id = native.randTensor(shape.map(Number));
        return new Tensor(id, shape);
    }

    static randn(shape: Shape): Tensor {
        const id = native.randnTensor(shape.map(Number));
        return new Tensor(id, shape);
    }

    // ---- Elementwise ops ----

    add(other: Tensor | number): Tensor {
        if (typeof other === 'number') {
            const s = Tensor.fromFloat32(new Float32Array([other]), [1]);
            return new Tensor(native.add(this._id, s._id));
        }
        return new Tensor(native.add(this._id, other._id));
    }

    sub(other: Tensor | number): Tensor {
        if (typeof other === 'number') {
            const s = Tensor.fromFloat32(new Float32Array([other]), [1]);
            return new Tensor(native.sub(this._id, s._id));
        }
        return new Tensor(native.sub(this._id, other._id));
    }

    mul(other: Tensor | number): Tensor {
        if (typeof other === 'number') {
            return new Tensor(native.mulScalar(this._id, other));
        }
        return new Tensor(native.mul(this._id, other._id));
    }

    neg(): Tensor {
        return new Tensor(native.neg(this._id));
    }

    exp(): Tensor {
        return new Tensor(native.expOp(this._id));
    }

    log(): Tensor {
        return new Tensor(native.logOp(this._id));
    }

    // ---- Activation ----

    relu(): Tensor {
        return new Tensor(native.relu(this._id));
    }

    sigmoid(): Tensor {
        // sigmoid(x) = 1 / (1 + exp(-x))
        const neg_x = this.neg();
        const exp_neg = neg_x.exp();
        const one_plus = exp_neg.add(1.0);
        const one = Tensor.ones([1]);
        return new Tensor(native.mul(one._id, one_plus._id)).log().neg().exp();
    }

    // ---- Reduction ----

    sum(dim?: number): Tensor {
        if (dim == null) {
            let result: Tensor = this;
            while (result.dims > 0) {
                result = new Tensor(native.sumOp(result._id, result.dims - 1));
            }
            return result;
        }

        return new Tensor(native.sumOp(this._id, dim));
    }

    mean(dim?: number): Tensor {
        if (dim == null) {
            let result: Tensor = this;
            while (result.dims > 0) {
                result = new Tensor(native.meanOp(result._id, result.dims - 1));
            }
            return result;
        }

        return new Tensor(native.meanOp(this._id, dim));
    }

    max(dim?: number): Tensor {
        if (dim == null) {
            let result: Tensor = this;
            while (result.dims > 0) {
                result = new Tensor(native.maxOp(result._id, result.dims - 1));
            }
            return result;
        }

        return new Tensor(native.maxOp(this._id, dim));
    }

    // ---- Layout ----

    view(...shape: number[]): Tensor {
        return new Tensor(native.view(this._id, shape.map(Number)));
    }

    permute(...dims: number[]): Tensor {
        return new Tensor(native.permute(this._id, dims.map(Number)));
    }

    contiguous(): Tensor {
        return new Tensor(native.contiguous(this._id));
    }

    // ---- Linear algebra ----

    matmul(other: Tensor): Tensor {
        return new Tensor(native.matmul(this._id, other._id));
    }

    // ---- Parameter management ----

    setRequiresGrad(requires: boolean): Tensor {
        native.setRequiresGrad(this._id, requires);
        return this;
    }

    zero_grad_(): void {
        native.zeroGrad([this._id]);
    }

    // Conv stubs for API compat
    conv1d(_weight: Tensor): Tensor { throw new Error('conv1d not implemented in native backend'); }
    conv2d(_weight: Tensor): Tensor { throw new Error('conv2d not implemented in native backend'); }
}

export type TensorLike = number | Tensor;
