export class IndexingError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'IndexingError';
    }
}

export type Shape = readonly number[];

export type Strides = readonly number[];

export type Index = readonly number[];

export type OutIndex = number[];

export type Storage = Float64Array;

export function indexToPosition(idx: Index, strides: Strides): number {
    let position = 0;
    for (let i = 0; i < idx.length; i++) {
        position += idx[i]! * strides[i]!;
    }
    return position;
}

export function toIndex(ordinal: number, shape: Shape, outIndex: OutIndex): void {
    let remaining = ordinal;
    for (let i = shape.length - 1; i >= 0; i--) {
        const dimSize = shape[i]!;
        outIndex[i] = remaining % dimSize;
        remaining = Math.floor(remaining / dimSize);
    }
}

export function shapeProduct(shape: Shape): number {
    let product = 1;
    for (const dim of shape) {
        product *= dim;
    }
    return product;
}

export function strides(shape: Shape): number[] {
    const result: number[] = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        result[i] = stride;
        stride *= shape[i]!;
    }
    return result;
}

export function createSharedStorage(size: number): Storage {
    const buffer = new SharedArrayBuffer(size * Float64Array.BYTES_PER_ELEMENT);
    return new Float64Array(buffer);
}

export class TensorData {
    readonly storage: Storage;
    readonly shape: Shape;
    readonly strides: Strides;
    readonly size: number;
    readonly dims: number;

    constructor(
        storage: Storage,
        shape: Shape,
        inputStrides?: Strides,
    ) {
        this.storage = storage;
        this.shape = shape;
        this.strides = inputStrides ?? strides(shape);
        this.size = shapeProduct(shape);
        this.dims = shape.length;

        if (this.strides.length !== this.dims) {
            throw new Error(
                `Strides length (${this.strides.length}) must match shape length (${this.dims})`
            );
        }
    }

    static zeros(shape: Shape): TensorData {
        const size = shapeProduct(shape);
        const storage = createSharedStorage(size);
        return new TensorData(storage, shape);
    }

    get(idx: Index): number {
        return this.storage[indexToPosition(idx, this.strides)]!;
    }

    set(idx: Index, value: number): void {
        this.storage[indexToPosition(idx, this.strides)] = value;
    }

    permute(...order: number[]): TensorData {
        if (order.length !== this.dims) {
            throw new Error(
                `Permutation length(${order.length}) must match number of dimensions (${this.dims})`
            );
        }

        const seen = new Set<number>();
        for (const i of order) {
            if (i < 0 || i >= this.dims) {
                throw new Error(`Invalid dimension index: ${i}`);
            }
            if (seen.has(i)) {
                throw new Error(`Duplicate dimension in permutation: ${i}`);
            }
            seen.add(i);
        }

        const newShape: number[] = new Array(this.dims);
        const newStrides: number[] = new Array(this.dims);

        for (let i = 0; i < this.dims; i++) {
            newShape[i] = this.shape[order[i]!]!;
            newStrides[i] = this.strides[order[i]!]!;
        }

        return new TensorData(this.storage, newShape, newStrides);
    }

    toString(): string {
        return `TensorData(shape=${JSON.stringify(this.shape)}, strides=${JSON.stringify(this.strides)})`;
    }
}

export function shapeBroadcast(shape1: Shape, shape2: Shape): number[] {
    const maxDims = Math.max(shape1.length, shape2.length);
    const result: number[] = new Array(maxDims);

    for (let i = 0; i < maxDims; i++) {
        const idx1 = shape1.length - 1 - i;
        const idx2 = shape2.length - 1 - i;

        const dim1 = idx1 >= 0 ? shape1[idx1]! : 1;
        const dim2 = idx2 >= 0 ? shape2[idx2]! : 1;

        if (dim1 == dim2) {
            result[maxDims - 1 - i] = dim1;
        } else if (dim1 === 1) {
            result[maxDims - 1 - i] = dim2;
        } else if (dim2 === 1) {
            result[maxDims - 1 - i] = dim1;
        } else {
            throw new IndexingError(
                `Cannot broadcast shapes [${shape1.join(', ')}] and [${shape2.join(', ')}]: ` + 
                `dimension mismatch at position ${-i - 1} (${dim1} vs ${dim2})`
            )
        }
    }

    return result;
}

export function broadcastIndex (
    bigIndex: Index,
    bigShape: Shape,
    shape: Shape,
    outIndex: OutIndex
): void {
    const offset = bigShape.length - shape.length;

    for (let i = 0; i < shape.length; i++) {
        const bigI = i + offset;

        if (shape[i] === 1) {
            outIndex[i] = 0
        } else {
            outIndex[i] = bigIndex[bigI]!;
        }
    }
}