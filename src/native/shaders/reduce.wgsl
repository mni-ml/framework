@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;

struct Params { outer: u32, dim_size: u32, inner: u32 }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn sum_along_dim_f32(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params.outer * params.inner;
    if (idx >= total) { return; }
    let o = idx / params.inner;
    let j = idx % params.inner;
    var sum_val: f32 = 0.0;
    for (var d: u32 = 0u; d < params.dim_size; d++) {
        sum_val += inp[(o * params.dim_size + d) * params.inner + j];
    }
    out[idx] = sum_val;
}

@compute @workgroup_size(256)
fn mean_along_dim_f32(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params.outer * params.inner;
    if (idx >= total) { return; }
    let o = idx / params.inner;
    let j = idx % params.inner;
    var sum_val: f32 = 0.0;
    for (var d: u32 = 0u; d < params.dim_size; d++) {
        sum_val += inp[(o * params.dim_size + d) * params.inner + j];
    }
    out[idx] = sum_val / f32(params.dim_size);
}

@compute @workgroup_size(256)
fn max_along_dim_f32(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params.outer * params.inner;
    if (idx >= total) { return; }
    let o = idx / params.inner;
    let j = idx % params.inner;
    var max_val: f32 = -1e30;
    for (var d: u32 = 0u; d < params.dim_size; d++) {
        max_val = max(max_val, inp[(o * params.dim_size + d) * params.inner + j]);
    }
    out[idx] = max_val;
}
