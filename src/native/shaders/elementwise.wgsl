@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;

struct Params { n: u32, scalar: f32 }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn add_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = a[i] + b[i]; }
}

@compute @workgroup_size(256)
fn sub_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = a[i] - b[i]; }
}

@compute @workgroup_size(256)
fn mul_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = a[i] * b[i]; }
}

@compute @workgroup_size(256)
fn neg_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = -a[i]; }
}

@compute @workgroup_size(256)
fn mul_scalar_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = a[i] * params.scalar; }
}

@compute @workgroup_size(256)
fn exp_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = exp(a[i]); }
}

@compute @workgroup_size(256)
fn log_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = log(a[i]); }
}

@compute @workgroup_size(256)
fn div_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = a[i] / b[i]; }
}

@compute @workgroup_size(256)
fn fill_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = params.scalar; }
}

@compute @workgroup_size(256)
fn copy_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = a[i]; }
}

@compute @workgroup_size(256)
fn lt_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = select(0.0, 1.0, a[i] < b[i]); }
}

@compute @workgroup_size(256)
fn eq_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = select(0.0, 1.0, abs(a[i] - b[i]) < 1e-6); }
}

@compute @workgroup_size(256)
fn gt_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = select(0.0, 1.0, a[i] > b[i]); }
}

@compute @workgroup_size(256)
fn pow_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = pow(a[i], params.scalar); }
}

@compute @workgroup_size(256)
fn sigmoid_forward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = 1.0 / (1.0 + exp(-a[i])); }
}

@compute @workgroup_size(256)
fn sigmoid_backward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = a[i] * b[i] * (1.0 - b[i]); } // a=grad, b=sigmoid_out
}
