import * as operators from './operators.js';

export const WORKGROUP_SIZE = 256;
const MAX_DIMS = 6;

// ---- Operation registries ----

export const UNARY_OPS: Record<string, string> = {
    neg:     'return -x;',
    id:      'return x;',
    sigmoid: 'let s = select(1.0 / (1.0 + exp(-x)), exp(x) / (1.0 + exp(x)), x < 0.0); return s;',
    relu:    'return max(x, 0.0);',
    exp:     'return exp(x);',
    log:     'return log(x);',
    inv:     'return 1.0 / x;',
};

export const BINARY_OPS: Record<string, string> = {
    add:     'return a + b;',
    mul:     'return a * b;',
    max:     'return max(a, b);',
    lt:      'return select(0.0, 1.0, a < b);',
    eq:      'return select(0.0, 1.0, abs(a - b) < 0.00001);',
    isClose: 'return select(0.0, 1.0, abs(a - b) < 0.01);',
};

// Map TypeScript operator functions to their registry key.
const unaryRegistry = new Map<Function, string>();
unaryRegistry.set(operators.neg,     'neg');
unaryRegistry.set(operators.id,      'id');
unaryRegistry.set(operators.sigmoid, 'sigmoid');
unaryRegistry.set(operators.relu,    'relu');
unaryRegistry.set(operators.exp,     'exp');
unaryRegistry.set(operators.log,     'log');
unaryRegistry.set(operators.inv,     'inv');

const binaryRegistry = new Map<Function, string>();
binaryRegistry.set(operators.add,     'add');
binaryRegistry.set(operators.mul,     'mul');
binaryRegistry.set(operators.max,     'max');
binaryRegistry.set(operators.lt,      'lt');
binaryRegistry.set(operators.eq,      'eq');
binaryRegistry.set(operators.isClose, 'isClose');

export function resolveUnaryOp(fn: Function): string {
    const name = unaryRegistry.get(fn);
    if (name) return name;
    throw new Error(`Unknown GPU unary op: ${fn.name || fn.toString().slice(0, 60)}`);
}

export function resolveBinaryOp(fn: Function): string {
    const name = binaryRegistry.get(fn);
    if (name) return name;
    throw new Error(`Unknown GPU binary op: ${fn.name || fn.toString().slice(0, 60)}`);
}

export const REDUCE_IDENTITY: Record<string, string> = {
    add: '0.0',
    mul: '1.0',
    max: '-1.0e+38',
};

// ---- WGSL utility: index helpers ported to WGSL ----
// Uses storage buffers for shape/stride arrays to avoid uniform alignment restrictions.

const WGSL_INDEX_HELPERS = `
const MAX_DIMS: u32 = ${MAX_DIMS}u;

fn toIndex(ordinal: u32, shape: array<u32, ${MAX_DIMS}>, dims: u32, out_idx: ptr<function, array<u32, ${MAX_DIMS}>>) {
    var remaining = ordinal;
    for (var i = i32(dims) - 1; i >= 0; i--) {
        let d = shape[i];
        (*out_idx)[i] = remaining % d;
        remaining = remaining / d;
    }
}

fn indexToPosition(idx: array<u32, ${MAX_DIMS}>, strd: array<u32, ${MAX_DIMS}>, dims: u32) -> u32 {
    var pos: u32 = 0u;
    for (var i: u32 = 0u; i < dims; i++) {
        pos += idx[i] * strd[i];
    }
    return pos;
}

fn broadcastIndex(
    bigIdx: array<u32, ${MAX_DIMS}>, bigDims: u32,
    smallShape: array<u32, ${MAX_DIMS}>, smallDims: u32,
    out_idx: ptr<function, array<u32, ${MAX_DIMS}>>
) {
    let off = bigDims - smallDims;
    for (var i: u32 = 0u; i < smallDims; i++) {
        let bigI = i + off;
        if (smallShape[i] == 1u) {
            (*out_idx)[i] = 0u;
        } else {
            (*out_idx)[i] = bigIdx[bigI];
        }
    }
}
`;

// ---- Shader template builders ----

/**
 * Aligned map: shapes & strides match, simple 1:1 element mapping.
 */
export function buildAlignedMapShader(opBody: string): string {
    return `
@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<f32>;

struct Params { size: u32 }
@group(0) @binding(2) var<uniform> params: Params;

fn apply(x: f32) -> f32 { ${opBody} }

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.size) { return; }
    out_data[i] = apply(in_data[i]);
}
`;
}

/**
 * Broadcast map: output and input may differ in shape.
 * Uses storage buffer for params to avoid WGSL uniform array alignment rules.
 */
export function buildBroadcastMapShader(opBody: string): string {
    return `
${WGSL_INDEX_HELPERS}

@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<f32>;

struct Params {
    out_size: u32,
    out_dims: u32,
    in_dims: u32,
    _pad: u32,
    out_shape:   array<u32, ${MAX_DIMS}>,
    out_strides: array<u32, ${MAX_DIMS}>,
    in_shape:    array<u32, ${MAX_DIMS}>,
    in_strides:  array<u32, ${MAX_DIMS}>,
}
@group(0) @binding(2) var<storage, read> params: Params;

fn apply(x: f32) -> f32 { ${opBody} }

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.out_size) { return; }

    var outIdx: array<u32, ${MAX_DIMS}>;
    toIndex(i, params.out_shape, params.out_dims, &outIdx);

    var inIdx: array<u32, ${MAX_DIMS}>;
    broadcastIndex(outIdx, params.out_dims, params.in_shape, params.in_dims, &inIdx);

    let outPos = indexToPosition(outIdx, params.out_strides, params.out_dims);
    let inPos  = indexToPosition(inIdx, params.in_strides, params.in_dims);
    out_data[outPos] = apply(in_data[inPos]);
}
`;
}

/**
 * Aligned zip: all three tensors share shape & strides.
 */
export function buildAlignedZipShader(opBody: string): string {
    return `
@group(0) @binding(0) var<storage, read> a_data: array<f32>;
@group(0) @binding(1) var<storage, read> b_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_data: array<f32>;

struct Params { size: u32 }
@group(0) @binding(3) var<uniform> params: Params;

fn apply(a: f32, b: f32) -> f32 { ${opBody} }

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.size) { return; }
    out_data[i] = apply(a_data[i], b_data[i]);
}
`;
}

/**
 * Broadcast zip: output, a, b may differ in shape.
 * Uses storage buffer for params to avoid WGSL uniform array alignment rules.
 */
export function buildBroadcastZipShader(opBody: string): string {
    return `
${WGSL_INDEX_HELPERS}

@group(0) @binding(0) var<storage, read> a_data: array<f32>;
@group(0) @binding(1) var<storage, read> b_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_data: array<f32>;

struct Params {
    out_size: u32,
    out_dims: u32,
    a_dims: u32,
    b_dims: u32,
    out_shape:   array<u32, ${MAX_DIMS}>,
    out_strides: array<u32, ${MAX_DIMS}>,
    a_shape:     array<u32, ${MAX_DIMS}>,
    a_strides:   array<u32, ${MAX_DIMS}>,
    b_shape:     array<u32, ${MAX_DIMS}>,
    b_strides:   array<u32, ${MAX_DIMS}>,
}
@group(0) @binding(3) var<storage, read> params: Params;

fn apply(a: f32, b: f32) -> f32 { ${opBody} }

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.out_size) { return; }

    var outIdx: array<u32, ${MAX_DIMS}>;
    toIndex(i, params.out_shape, params.out_dims, &outIdx);

    var aIdx: array<u32, ${MAX_DIMS}>;
    broadcastIndex(outIdx, params.out_dims, params.a_shape, params.a_dims, &aIdx);

    var bIdx: array<u32, ${MAX_DIMS}>;
    broadcastIndex(outIdx, params.out_dims, params.b_shape, params.b_dims, &bIdx);

    let outPos = indexToPosition(outIdx, params.out_strides, params.out_dims);
    let aPos   = indexToPosition(aIdx, params.a_strides, params.a_dims);
    let bPos   = indexToPosition(bIdx, params.b_strides, params.b_dims);
    out_data[outPos] = apply(a_data[aPos], b_data[bPos]);
}
`;
}

/**
 * Sum practice: block-level partial sums using shared memory.
 * Input: array of length size. Output: array of length ceil(size / WORKGROUP_SIZE).
 * Each workgroup sums WORKGROUP_SIZE contiguous elements into one output cell.
 */
export function buildSumPracticeShader(): string {
    return `
const BLOCK_DIM: u32 = ${WORKGROUP_SIZE}u;
var<workgroup> sdata: array<f32, ${WORKGROUP_SIZE}>;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params { size: u32 }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3u,
) {
    let global_idx = wid.x * BLOCK_DIM + tid;

    if (global_idx < params.size) {
        sdata[tid] = a[global_idx];
    } else {
        sdata[tid] = 0.0;
    }
    workgroupBarrier();

    for (var stride = BLOCK_DIM / 2u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            sdata[tid] = sdata[tid] + sdata[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        result[wid.x] = sdata[0];
    }
}
`;
}

/**
 * General reduce along one dimension.
 * One workgroup per output element. Threads cooperatively reduce
 * the reduction dimension using shared memory.
 * Uses storage buffer for params to avoid WGSL uniform array alignment rules.
 */
export function buildReduceShader(opBody: string, identity: string): string {
    return `
${WGSL_INDEX_HELPERS}

const BLOCK_DIM: u32 = ${WORKGROUP_SIZE}u;
var<workgroup> sdata: array<f32, ${WORKGROUP_SIZE}>;

@group(0) @binding(0) var<storage, read> a_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<f32>;

struct Params {
    out_size: u32,
    out_dims: u32,
    a_dims: u32,
    reduce_dim: u32,
    reduce_dim_size: u32,
    reduce_stride: u32,
    _pad0: u32,
    _pad1: u32,
    out_shape:   array<u32, ${MAX_DIMS}>,
    out_strides: array<u32, ${MAX_DIMS}>,
    a_shape:     array<u32, ${MAX_DIMS}>,
    a_strides:   array<u32, ${MAX_DIMS}>,
}
@group(0) @binding(2) var<storage, read> params: Params;

fn apply(a: f32, b: f32) -> f32 { ${opBody} }

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3u,
) {
    let out_idx = wid.x;
    if (out_idx >= params.out_size) { return; }

    var outMI: array<u32, ${MAX_DIMS}>;
    toIndex(out_idx, params.out_shape, params.out_dims, &outMI);

    var aIdx: array<u32, ${MAX_DIMS}>;
    for (var d: u32 = 0u; d < params.a_dims; d++) {
        aIdx[d] = outMI[d];
    }
    aIdx[params.reduce_dim] = 0u;
    let base_pos = indexToPosition(aIdx, params.a_strides, params.a_dims);

    var local_acc: f32 = ${identity};
    for (var j = tid; j < params.reduce_dim_size; j += BLOCK_DIM) {
        local_acc = apply(local_acc, a_data[base_pos + j * params.reduce_stride]);
    }
    sdata[tid] = local_acc;
    workgroupBarrier();

    for (var s = BLOCK_DIM / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            sdata[tid] = apply(sdata[tid], sdata[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let outPos = indexToPosition(outMI, params.out_strides, params.out_dims);
        out_data[outPos] = sdata[0];
    }
}
`;
}
