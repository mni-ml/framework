@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read_write> mean_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> rstd_out: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read> gamma: array<f32>;
@group(0) @binding(5) var<storage, read> beta: array<f32>;

struct Params { n: u32, c: u32, eps: f32 }
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(256)
fn layernorm_forward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let row = gid.x;
    if (row >= params.n) { return; }
    let c = params.c;
    let base = row * c;

    var sum_val: f32 = 0.0;
    for (var j: u32 = 0u; j < c; j++) { sum_val += x[base + j]; }
    let mean_val = sum_val / f32(c);

    var var_val: f32 = 0.0;
    for (var j: u32 = 0u; j < c; j++) {
        let diff = x[base + j] - mean_val;
        var_val += diff * diff;
    }
    let rstd = 1.0 / sqrt(var_val / f32(c) + params.eps);

    for (var j: u32 = 0u; j < c; j++) {
        out[base + j] = gamma[j] * (x[base + j] - mean_val) * rstd + beta[j];
    }
    mean_out[row] = mean_val;
    rstd_out[row] = rstd;
}

@group(0) @binding(0) var<storage, read_write> dx: array<f32>;
@group(0) @binding(1) var<storage, read_write> dgamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> dbeta: array<f32>;
@group(0) @binding(3) var<storage, read> dy_bwd: array<f32>;
@group(0) @binding(4) var<storage, read> x_bwd: array<f32>;
@group(0) @binding(5) var<storage, read> mean_bwd: array<f32>;
@group(0) @binding(6) var<storage, read> rstd_bwd: array<f32>;
@group(0) @binding(7) var<storage, read> gamma_bwd: array<f32>;

struct ParamsBwd { n: u32, c: u32 }
@group(0) @binding(8) var<uniform> params_bwd: ParamsBwd;

@compute @workgroup_size(256)
fn layernorm_backward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let row = gid.x;
    if (row >= params_bwd.n) { return; }
    let c = params_bwd.c;
    let base = row * c;
    let m = mean_bwd[row];
    let r = rstd_bwd[row];

    var dot_dy_xhat: f32 = 0.0;
    var dot_dy_val: f32 = 0.0;
    for (var j: u32 = 0u; j < c; j++) {
        let xhat = (x_bwd[base + j] - m) * r;
        dot_dy_xhat += dy_bwd[base + j] * gamma_bwd[j] * xhat;
        dot_dy_val += dy_bwd[base + j] * gamma_bwd[j];
    }
    let inv_c = 1.0 / f32(c);
    for (var j: u32 = 0u; j < c; j++) {
        let xhat = (x_bwd[base + j] - m) * r;
        dx[base + j] = r * (dy_bwd[base + j] * gamma_bwd[j] - inv_c * (dot_dy_val + xhat * dot_dy_xhat));
    }
}
