@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;

struct Params { outer: u32, dim_size: u32, inner: u32 }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn softmax_forward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let row = gid.x;
    let total = params.outer * params.inner;
    if (row >= total) { return; }
    let o = row / params.inner;
    let j = row % params.inner;

    var max_val: f32 = -1e30;
    for (var d: u32 = 0u; d < params.dim_size; d++) {
        max_val = max(max_val, x[(o * params.dim_size + d) * params.inner + j]);
    }

    var sum_exp: f32 = 0.0;
    for (var d: u32 = 0u; d < params.dim_size; d++) {
        let e = exp(x[(o * params.dim_size + d) * params.inner + j] - max_val);
        out[(o * params.dim_size + d) * params.inner + j] = e;
        sum_exp += e;
    }

    let inv_sum = 1.0 / sum_exp;
    for (var d: u32 = 0u; d < params.dim_size; d++) {
        out[(o * params.dim_size + d) * params.inner + j] *= inv_sum;
    }
}

@group(0) @binding(0) var<storage, read_write> dx: array<f32>;
@group(0) @binding(1) var<storage, read> dy_sm: array<f32>;
@group(0) @binding(2) var<storage, read> out_sm: array<f32>;

struct ParamsBwd { outer: u32, dim_size: u32, inner: u32 }
@group(0) @binding(3) var<uniform> params_bwd: ParamsBwd;

@compute @workgroup_size(256)
fn softmax_backward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let row = gid.x;
    let total = params_bwd.outer * params_bwd.inner;
    if (row >= total) { return; }
    let o = row / params_bwd.inner;
    let j = row % params_bwd.inner;

    var dot: f32 = 0.0;
    for (var d: u32 = 0u; d < params_bwd.dim_size; d++) {
        let pos = (o * params_bwd.dim_size + d) * params_bwd.inner + j;
        dot += dy_sm[pos] * out_sm[pos];
    }
    for (var d: u32 = 0u; d < params_bwd.dim_size; d++) {
        let pos = (o * params_bwd.dim_size + d) * params_bwd.inner + j;
        dx[pos] = out_sm[pos] * (dy_sm[pos] - dot);
    }
}
