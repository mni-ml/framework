import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname_f = dirname(fileURLToPath(import.meta.url));

// Check whether CUDA drivers are available on this system.
// The CUDA native binary will abort the process on load if the driver library is missing,
// so we must check before attempting require().
function hasCudaDriver(): boolean {
    if (process.platform === 'linux') {
        const searchPaths = [
            '/usr/lib/x86_64-linux-gnu',
            '/usr/lib64',
            '/usr/local/cuda/lib64',
            '/usr/local/cuda/targets/x86_64-linux/lib',
            '/usr/lib',
        ];
        const ldPath = process.env['LD_LIBRARY_PATH'] ?? '';
        if (ldPath) searchPaths.push(...ldPath.split(':').filter(Boolean));
        for (const dir of searchPaths) {
            if (existsSync(join(dir, 'libcuda.so')) || existsSync(join(dir, 'libcuda.so.1'))) {
                return true;
            }
        }
        return false;
    }
    if (process.platform === 'win32') {
        const system32 = join(process.env['SystemRoot'] ?? 'C:\\Windows', 'System32');
        if (existsSync(join(system32, 'nvcuda.dll'))) return true;
        const pathDirs = (process.env['PATH'] ?? '').split(';').filter(Boolean);
        for (const dir of pathDirs) {
            if (existsSync(join(dir, 'nvcuda.dll'))) return true;
        }
        return false;
    }
    return false;
}

// Priority: CUDA (if driver detected) > WebGPU > CPU.
// CUDA entries are only included when a CUDA driver is detected to prevent process abort.
function getPlatformPackages(): Record<string, string[]> {
    const cuda = hasCudaDriver();
    return {
        'darwin-arm64':  ['@mni-ml/framework-darwin-arm64-webgpu', '@mni-ml/framework-darwin-arm64'],
        'darwin-x64':    ['@mni-ml/framework-darwin-x64-webgpu',   '@mni-ml/framework-darwin-x64'],
        'linux-x64':     [
            ...(cuda ? ['@mni-ml/framework-linux-x64-gnu-cuda'] : []),
            '@mni-ml/framework-linux-x64-gnu-webgpu',
            '@mni-ml/framework-linux-x64-gnu',
        ],
        'linux-arm64':   ['@mni-ml/framework-linux-arm64-gnu'],
        'win32-x64':     [
            ...(cuda ? ['@mni-ml/framework-win32-x64-msvc-cuda'] : []),
            '@mni-ml/framework-win32-x64-msvc-webgpu',
            '@mni-ml/framework-win32-x64-msvc',
        ],
    };
}

function getLocalSuffixes(): Record<string, string[]> {
    const cuda = hasCudaDriver();
    return {
        'darwin-arm64': ['darwin-arm64-webgpu', 'darwin-arm64'],
        'darwin-x64':   ['darwin-x64-webgpu',   'darwin-x64'],
        'linux-x64':    [
            ...(cuda ? ['linux-x64-gnu-cuda'] : []),
            'linux-x64-gnu-webgpu',
            'linux-x64-gnu',
        ],
        'linux-arm64':  ['linux-arm64-gnu'],
        'win32-x64':    [
            ...(cuda ? ['win32-x64-msvc-cuda'] : []),
            'win32-x64-msvc-webgpu',
            'win32-x64-msvc',
        ],
    };
}

function loadNative() {
    const require = createRequire(import.meta.url);
    const platform = process.platform;
    const arch = process.arch;
    const key = `${platform}-${arch}`;

    // 1. Try prebuilt platform packages (GPU first, then CPU fallback)
    const candidates_pkg = getPlatformPackages()[key] ?? [];
    for (const pkgName of candidates_pkg) {
        try { return require(pkgName); } catch {}
    }

    // 2. Fall back to a local .node file (dev builds / build-from-source)
    //    GPU binaries are tried first, then CPU fallback.
    const suffixes = getLocalSuffixes()[key] ?? [key];
    const ext = platform === 'win32' ? 'dll' : platform === 'darwin' ? 'dylib' : 'so';
    const candidates_file: string[] = [];
    for (const suffix of suffixes) {
        candidates_file.push(join(__dirname_f, '..', 'src', 'native', `mni-framework-native.${suffix}.node`));
    }
    candidates_file.push(join(__dirname_f, '..', 'src', 'native', 'target', 'release', `libmni_framework_native.${ext}`));
    for (const p of candidates_file) {
        if (existsSync(p)) {
            return require(p);
        }
    }

    const hint = candidates_pkg.length > 0
        ? `\n  Install prebuilt: npm install ${candidates_pkg[candidates_pkg.length - 1]}\n  For CUDA: npm install ${candidates_pkg[0]}\n  Or build from source: cd src/native && cargo build --release`
        : `\n  Build from source: cd src/native && cargo build --release`;
    throw new Error(
        `@mni-ml/framework: native addon not found for ${platform}-${arch}.${hint}`
    );
}

export const native: any = loadNative();

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

    // Compatibility stubs for old code that accesses .data.storage or .history
    get data(): { storage: Float32Array } {
        return { storage: this.toFloat32() };
    }
    set history(_v: any) { /* no-op: autograd is in Rust */ }
    get history(): null { return null; }

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

    // ---- Comparison (returns 0.0/1.0 tensors, no gradient) ----

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
