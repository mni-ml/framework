@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;

struct Params { n: u32 }
@group(0) @binding(3) var<uniform> params: Params;

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const COEFF: f32 = 0.044715;

@compute @workgroup_size(256)
fn gelu_forward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) {
        let xi = x[i];
        let inner = SQRT_2_OVER_PI * (xi + COEFF * xi * xi * xi);
        out[i] = 0.5 * xi * (1.0 + tanh(inner));
    }
}

@compute @workgroup_size(256)
fn gelu_backward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) {
        let xi = x[i];
        let x2 = xi * xi;
        let inner = SQRT_2_OVER_PI * (xi + COEFF * xi * x2);
        let th = tanh(inner);
        let sech2 = 1.0 - th * th;
        let d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x2);
        let grad = 0.5 * (1.0 + th) + 0.5 * xi * sech2 * d_inner;
        out[i] = dy[i] * grad;
    }
}

@compute @workgroup_size(256)
fn relu_forward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = max(x[i], 0.0); }
}

@compute @workgroup_size(256)
fn relu_backward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) { out[i] = select(0.0, dy[i], x[i] > 0.0); }
}
