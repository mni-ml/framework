import { Worker } from 'node:worker_threads';
import { cpus } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { existsSync } from 'node:fs';

import type {
    Storage,
    Shape,
    Strides,
} from './tensor_data.js';

import {
    indexToPosition,
    toIndex,
    shapeProduct,
    broadcastIndex,
} from './tensor_data.js';

const NUM_WORKERS = Math.max(1, cpus().length);

// Parallelism is across independent output elements, not within a single
// reduction, so both paths produce bitwise-identical results for the same fn.
const PARALLEL_THRESHOLD = 4096;

function shapesEqual(a: Shape, b: Shape): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

function stridesEqual(a: Strides, b: Strides): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

class WorkerPool {
    private workers: Worker[];
    private syncBuffer: SharedArrayBuffer;
    private syncArray: Int32Array;

    constructor(numWorkers: number) {
        this.syncBuffer = new SharedArrayBuffer(numWorkers * Int32Array.BYTES_PER_ELEMENT);
        this.syncArray = new Int32Array(this.syncBuffer);

        // .ts in dev (Node 25+ type-stripping), .js after tsc build
        const currentDir = dirname(fileURLToPath(import.meta.url));
        const tsPath = join(currentDir, 'fast_ops_worker.ts');
        const jsPath = join(currentDir, 'fast_ops_worker.js');
        const workerPath = existsSync(tsPath) ? tsPath : jsPath;

        this.workers = Array.from({ length: numWorkers }, (_, id) => {
            const w = new Worker(workerPath, {
                workerData: { workerId: id, syncBuffer: this.syncBuffer },
            });
            w.on('error', (err) => {
                console.error(`[fast_ops] Worker ${id} error:`, err);
            });
            return w;
        });
    }

    /** Split [0, size) across workers; block until all signal via Atomics. */
    parallelFor(
        size: number,
        taskFactory: (start: number, end: number) => object,
    ): void {
        const numWorkers = this.workers.length;
        const chunkSize = Math.ceil(size / numWorkers);

        for (let i = 0; i < numWorkers; i++) {
            Atomics.store(this.syncArray, i, 0);
        }

        for (let i = 0; i < numWorkers; i++) {
            const start = i * chunkSize;
            const end = Math.min(start + chunkSize, size);
            if (start >= size) {
                Atomics.store(this.syncArray, i, 1);
                continue;
            }
            this.workers[i]!.postMessage(taskFactory(start, end));
        }

        for (let i = 0; i < numWorkers; i++) {
            while (Atomics.load(this.syncArray, i) === 0) {
                Atomics.wait(this.syncArray, i, 0, 30000);
            }
        }
    }

    terminate(): void {
        for (const w of this.workers) {
            w.terminate();
        }
    }
}

let _pool: WorkerPool | null | undefined = undefined;

// SharedArrayBuffer can't cross Jest's VM context boundary to real workers.
const PARALLEL_DISABLED = typeof process !== 'undefined' &&
    (process.env['JEST_WORKER_ID'] !== undefined ||
     process.env['TSTORCH_DISABLE_PARALLEL'] !== undefined);

function getPool(): WorkerPool | null {
    if (PARALLEL_DISABLED) return null;
    if (_pool === undefined) {
        try {
            _pool = new WorkerPool(NUM_WORKERS);
        } catch {
            _pool = null;
        }
    }
    return _pool;
}

/** Terminate the worker pool (call in test teardown so Node can exit). */
export function destroyPool(): void {
    if (_pool) {
        _pool.terminate();
        _pool = undefined;
    }
}

function isShared(storage: Storage): boolean {
    return storage.buffer instanceof SharedArrayBuffer;
}

/**
 * Parallel tensor map. Optimizations: main loop in parallel when
 * size >= PARALLEL_THRESHOLD; stride-aligned fast path avoids all indexing.
 *
 * `fn` must be a pure function (no captured variables) -- workers reconstruct
 * it from `fn.toString()` via `new Function()`.
 */
export function fastTensorMap(
    fn: (x: number) => number,
): (
    outStorage: Storage,
    outShape: Shape,
    outStrides: Strides,
    inStorage: Storage,
    inShape: Shape,
    inStrides: Strides,
) => void {
    return (
        outStorage: Storage,
        outShape: Shape,
        outStrides: Strides,
        inStorage: Storage,
        inShape: Shape,
        inStrides: Strides,
    ): void => {
        const size = shapeProduct(outShape);
        const aligned =
            shapesEqual(outShape, inShape) && stridesEqual(outStrides, inStrides);

        const pool = getPool();
        if (
            pool &&
            size >= PARALLEL_THRESHOLD &&
            isShared(outStorage) &&
            isShared(inStorage)
        ) {
            const fnSource = fn.toString();

            pool.parallelFor(size, (start, end) => ({
                type: 'map',
                fnSource,
                start,
                end,
                outBuffer: outStorage.buffer as SharedArrayBuffer,
                outShape: Array.from(outShape),
                outStrides: Array.from(outStrides),
                inBuffer: inStorage.buffer as SharedArrayBuffer,
                inShape: Array.from(inShape),
                inStrides: Array.from(inStrides),
                aligned,
            }));
            return;
        }

        if (aligned) {
            for (let i = 0; i < size; i++) {
                outStorage[i] = fn(inStorage[i]!);
            }
            return;
        }

        const outIndex: number[] = new Array(outShape.length).fill(0);
        const inIndex: number[] = new Array(inShape.length).fill(0);

        for (let ordinal = 0; ordinal < size; ordinal++) {
            toIndex(ordinal, outShape, outIndex);
            broadcastIndex(outIndex, outShape, inShape, inIndex);

            const inPos = indexToPosition(inIndex, inStrides);
            const outPos = indexToPosition(outIndex, outStrides);
            outStorage[outPos] = fn(inStorage[inPos]!);
        }
    };
}

/**
 * Parallel tensor zip. Same optimizations and pure-function constraint
 * as fastTensorMap; stride-aligned when all three tensors match.
 */
export function fastTensorZip(
    fn: (a: number, b: number) => number,
): (
    outStorage: Storage,
    outShape: Shape,
    outStrides: Strides,
    aStorage: Storage,
    aShape: Shape,
    aStrides: Strides,
    bStorage: Storage,
    bShape: Shape,
    bStrides: Strides,
) => void {
    return (
        outStorage: Storage,
        outShape: Shape,
        outStrides: Strides,
        aStorage: Storage,
        aShape: Shape,
        aStrides: Strides,
        bStorage: Storage,
        bShape: Shape,
        bStrides: Strides,
    ): void => {
        const size = shapeProduct(outShape);
        const aligned =
            shapesEqual(outShape, aShape) &&
            shapesEqual(outShape, bShape) &&
            stridesEqual(outStrides, aStrides) &&
            stridesEqual(outStrides, bStrides);

        const pool = getPool();
        if (
            pool &&
            size >= PARALLEL_THRESHOLD &&
            isShared(outStorage) &&
            isShared(aStorage) &&
            isShared(bStorage)
        ) {
            const fnSource = fn.toString();

            pool.parallelFor(size, (start, end) => ({
                type: 'zip',
                fnSource,
                start,
                end,
                outBuffer: outStorage.buffer as SharedArrayBuffer,
                outShape: Array.from(outShape),
                outStrides: Array.from(outStrides),
                aBuffer: aStorage.buffer as SharedArrayBuffer,
                aShape: Array.from(aShape),
                aStrides: Array.from(aStrides),
                bBuffer: bStorage.buffer as SharedArrayBuffer,
                bShape: Array.from(bShape),
                bStrides: Array.from(bStrides),
                aligned,
            }));
            return;
        }

        if (aligned) {
            for (let i = 0; i < size; i++) {
                outStorage[i] = fn(aStorage[i]!, bStorage[i]!);
            }
            return;
        }

        const outIndex: number[] = new Array(outShape.length).fill(0);
        const aIndex: number[] = new Array(aShape.length).fill(0);
        const bIndex: number[] = new Array(bShape.length).fill(0);

        for (let ordinal = 0; ordinal < size; ordinal++) {
            toIndex(ordinal, outShape, outIndex);
            broadcastIndex(outIndex, outShape, aShape, aIndex);
            broadcastIndex(outIndex, outShape, bShape, bIndex);

            const aPos = indexToPosition(aIndex, aStrides);
            const bPos = indexToPosition(bIndex, bStrides);
            const outPos = indexToPosition(outIndex, outStrides);
            outStorage[outPos] = fn(aStorage[aPos]!, bStorage[bPos]!);
        }
    };
}

/**
 * Parallel tensor reduce. Outer loop (output elements) in parallel; inner
 * reduction is sequential per element using stride-stepping. Same
 * pure-function constraint as fastTensorMap.
 */
export function fastTensorReduce(
    fn: (acc: number, x: number) => number,
): (
    outStorage: Storage,
    outShape: Shape,
    outStrides: Strides,
    aStorage: Storage,
    aShape: Shape,
    aStrides: Strides,
    reduceDim: number,
) => void {
    return (
        outStorage: Storage,
        outShape: Shape,
        outStrides: Strides,
        aStorage: Storage,
        aShape: Shape,
        aStrides: Strides,
        reduceDim: number,
    ): void => {
        const outSize = shapeProduct(outShape);
        const reduceDimSize = aShape[reduceDim]!;
        const reduceStride = aStrides[reduceDim]!;

        const pool = getPool();
        if (
            pool &&
            outSize >= PARALLEL_THRESHOLD &&
            isShared(outStorage) &&
            isShared(aStorage)
        ) {
            const fnSource = fn.toString();

            pool.parallelFor(outSize, (start, end) => ({
                type: 'reduce',
                fnSource,
                start,
                end,
                outBuffer: outStorage.buffer as SharedArrayBuffer,
                outShape: Array.from(outShape),
                outStrides: Array.from(outStrides),
                inBuffer: aStorage.buffer as SharedArrayBuffer,
                inShape: Array.from(aShape),
                inStrides: Array.from(aStrides),
                reduceDim,
                reduceDimSize,
            }));
            return;
        }

        const outIndex: number[] = new Array(outShape.length).fill(0);
        const aIndex: number[] = new Array(aShape.length).fill(0);

        for (let ordinal = 0; ordinal < outSize; ordinal++) {
            toIndex(ordinal, outShape, outIndex);

            const outPos = indexToPosition(outIndex, outStrides);

            for (let i = 0; i < outShape.length; i++) {
                aIndex[i] = outIndex[i]!;
            }
            aIndex[reduceDim] = 0;
            let aPos = indexToPosition(aIndex, aStrides);

            let acc = aStorage[aPos]!;
            for (let j = 1; j < reduceDimSize; j++) {
                aPos += reduceStride;
                acc = fn(acc, aStorage[aPos]!);
            }

            outStorage[outPos] = acc;
        }
    };
}
