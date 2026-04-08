@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;

struct Params {
    n: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bc1: f32,  // 1 / (1 - beta1^t)
    bc2: f32,  // 1 / (1 - beta2^t)
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn adamw_step_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.n) { return; }

    let g = grad[i];
    m[i] = params.beta1 * m[i] + (1.0 - params.beta1) * g;
    v[i] = params.beta2 * v[i] + (1.0 - params.beta2) * g * g;

    let m_hat = m[i] * params.bc1;
    let v_hat = v[i] * params.bc2;

    param[i] = param[i] * (1.0 - params.lr * params.weight_decay) - params.lr * m_hat / (sqrt(v_hat) + params.eps);
}
