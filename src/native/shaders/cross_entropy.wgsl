@group(0) @binding(0) var<storage, read_write> loss: array<f32>;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;

struct Params { batch: u32, vocab: u32 }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn cross_entropy_forward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let b = gid.x;
    if (b >= params.batch) { return; }
    let base = b * params.vocab;
    let target = targets[b];

    var max_val: f32 = -1e30;
    for (var v: u32 = 0u; v < params.vocab; v++) {
        max_val = max(max_val, logits[base + v]);
    }
    var sum_exp: f32 = 0.0;
    for (var v: u32 = 0u; v < params.vocab; v++) {
        sum_exp += exp(logits[base + v] - max_val);
    }
    loss[b] = -(logits[base + target] - max_val - log(sum_exp));
}

@group(0) @binding(0) var<storage, read_write> dlogits: array<f32>;
@group(0) @binding(1) var<storage, read> logits_bwd: array<f32>;
@group(0) @binding(2) var<storage, read> targets_bwd: array<u32>;
@group(0) @binding(3) var<storage, read> grad_scale: array<f32>;

struct ParamsBwd { batch: u32, vocab: u32 }
@group(0) @binding(4) var<uniform> params_bwd: ParamsBwd;

@compute @workgroup_size(256)
fn cross_entropy_backward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params_bwd.batch * params_bwd.vocab;
    if (idx >= total) { return; }
    let b = idx / params_bwd.vocab;
    let v = idx % params_bwd.vocab;
    let base = b * params_bwd.vocab;
    let target = targets_bwd[b];

    var max_val: f32 = -1e30;
    for (var vi: u32 = 0u; vi < params_bwd.vocab; vi++) {
        max_val = max(max_val, logits_bwd[base + vi]);
    }
    var sum_exp: f32 = 0.0;
    for (var vi: u32 = 0u; vi < params_bwd.vocab; vi++) {
        sum_exp += exp(logits_bwd[base + vi] - max_val);
    }
    let softmax_val = exp(logits_bwd[base + v] - max_val) / sum_exp;
    let indicator = select(0.0, 1.0, v == target);
    dlogits[idx] = (softmax_val - indicator) * grad_scale[0] / f32(params_bwd.batch);
}
