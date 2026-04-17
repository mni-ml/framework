export let native: any;

export function _setNative(backend: any) {
    native = backend;
}

export type Shape = number[];

export class Tensor {
    readonly _id: number;
    private _shape: Shape;

    constructor(id: number, shape?: Shape) {
        this._id = id;
        this._shape = shape ?? native.tensorShape(id).map(Number);
    }

    get shape(): Shape { return this._shape; }
    get size(): number { return this._shape.reduce((a: number, b: number) => a * b, 1); }
    get dims(): number { return this._shape.length; }

    get data(): { storage: Float32Array } {
        return { storage: this.toFloat32() };
    }
    set history(_v: any) { /* no-op: autograd is in native backend */ }
    get history(): null { return null; }

    // ---- Gradient ----

    get grad(): Tensor | null {
        const gid = native.getGrad(this._id);
        if (gid == null) return null;
        return new Tensor(gid);
    }
    set grad(_v: Tensor | null) { /* managed by backend */ }

    backward(): void {
        native.backward(this._id);
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

    sin(): Tensor {
        return new Tensor(native.sinOp(this._id));
    }

    cos(): Tensor {
        return new Tensor(native.cosOp(this._id));
    }

    sqrt(): Tensor {
        return new Tensor(native.sqrtOp(this._id));
    }

    // ---- Activation ----

    relu(): Tensor {
        return new Tensor(native.relu(this._id));
    }

    sigmoid(): Tensor {
        return new Tensor(native.sigmoid(this._id));
    }

    // ---- Reduction ----

    sum(dim?: number): Tensor {
        if (dim === undefined) return new Tensor(native.sumAll(this._id));
        return new Tensor(native.sumOp(this._id, dim));
    }

    mean(dim?: number): Tensor {
        if (dim === undefined) return new Tensor(native.meanAll(this._id));
        return new Tensor(native.meanOp(this._id, dim));
    }

    max(dim: number): Tensor {
        return new Tensor(native.maxOp(this._id, dim));
    }

    // ---- Comparison ----

    lt(other: Tensor): Tensor {
        return new Tensor(native.lt(this._id, other._id));
    }

    eq(other: Tensor): Tensor {
        return new Tensor(native.eqOp(this._id, other._id));
    }

    gt(other: Tensor): Tensor {
        return new Tensor(native.gt(this._id, other._id));
    }

    isClose(other: Tensor, tol: number = 1e-5): Tensor {
        return new Tensor(native.isClose(this._id, other._id, tol));
    }

    // ---- Elementwise ----

    div(other: Tensor | number): Tensor {
        if (typeof other === 'number') {
            return this.mul(1.0 / other);
        }
        return new Tensor(native.div(this._id, other._id));
    }

    pow(exponent: number): Tensor {
        return new Tensor(native.powOp(this._id, exponent));
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

    // ---- Convolution ----

    conv1d(weight: Tensor, stride: number = 1, padding: number = 0): Tensor {
        return new Tensor(native.conv1DForward(this._id, weight._id, stride, padding));
    }

    conv2d(weight: Tensor, stride: number = 1, padding: number = 0): Tensor {
        return new Tensor(native.conv2DForward(this._id, weight._id, stride, padding));
    }

    // ---- Utility ----

    clone(): Tensor {
        const data = this.toFloat32();
        return Tensor.fromFloat32(new Float32Array(data), [...this._shape]);
    }

    detach(): Tensor {
        const data = this.toFloat32();
        return Tensor.fromFloat32(new Float32Array(data), [...this._shape]);
    }

    toString(): string {
        const data = this.toFloat32();
        const shapeStr = `[${this._shape.join(', ')}]`;
        if (data.length <= 10) {
            return `Tensor(${shapeStr}, [${Array.from(data).map(v => v.toFixed(4)).join(', ')}])`;
        }
        const first = Array.from(data.slice(0, 5)).map(v => v.toFixed(4)).join(', ');
        const last = Array.from(data.slice(-3)).map(v => v.toFixed(4)).join(', ');
        return `Tensor(${shapeStr}, [${first}, ..., ${last}])`;
    }

    // ---- Parameter management ----

    setRequiresGrad(requires: boolean): Tensor {
        native.setRequiresGrad(this._id, requires);
        return this;
    }
}

export type TensorLike = number | Tensor;
