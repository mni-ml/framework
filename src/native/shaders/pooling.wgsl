@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;

struct PoolParams { N: u32, C: u32, H: u32, W: u32, kH: u32, kW: u32, H_out: u32, W_out: u32 }
@group(0) @binding(2) var<uniform> params: PoolParams;

@compute @workgroup_size(256)
fn avgpool2d_forward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params.N * params.C * params.H_out * params.W_out;
    if (idx >= total) { return; }
    let ow = idx % params.W_out;
    let oh = (idx / params.W_out) % params.H_out;
    let c = (idx / (params.W_out * params.H_out)) % params.C;
    let n = idx / (params.W_out * params.H_out * params.C);

    var sum_val: f32 = 0.0;
    var count: u32 = 0u;
    for (var kh: u32 = 0u; kh < params.kH; kh++) {
        for (var kw: u32 = 0u; kw < params.kW; kw++) {
            let ih = oh * params.kH + kh;
            let iw = ow * params.kW + kw;
            if (ih < params.H && iw < params.W) {
                sum_val += inp[n * params.C * params.H * params.W + c * params.H * params.W + ih * params.W + iw];
                count++;
            }
        }
    }
    out[idx] = sum_val / f32(count);
}

@compute @workgroup_size(256)
fn avgpool2d_backward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params.N * params.C * params.H * params.W;
    if (idx >= total) { return; }
    let iw = idx % params.W;
    let ih = (idx / params.W) % params.H;
    let c = (idx / (params.W * params.H)) % params.C;
    let n = idx / (params.W * params.H * params.C);

    let oh = ih / params.kH;
    let ow = iw / params.kW;
    if (oh < params.H_out && ow < params.W_out) {
        let inv = 1.0 / f32(params.kH * params.kW);
        out[idx] = inp[n * params.C * params.H_out * params.W_out + c * params.H_out * params.W_out + oh * params.W_out + ow] * inv;
    } else {
        out[idx] = 0.0;
    }
}

@group(0) @binding(0) var<storage, read_write> max_out: array<f32>;
@group(0) @binding(1) var<storage, read_write> argmax_out: array<u32>;
@group(0) @binding(2) var<storage, read> max_inp: array<f32>;

struct MaxPoolParams { N: u32, C: u32, H: u32, W: u32, kH: u32, kW: u32, H_out: u32, W_out: u32 }
@group(0) @binding(3) var<uniform> max_params: MaxPoolParams;

@compute @workgroup_size(256)
fn maxpool2d_forward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = max_params.N * max_params.C * max_params.H_out * max_params.W_out;
    if (idx >= total) { return; }
    let ow = idx % max_params.W_out;
    let oh = (idx / max_params.W_out) % max_params.H_out;
    let c = (idx / (max_params.W_out * max_params.H_out)) % max_params.C;
    let n = idx / (max_params.W_out * max_params.H_out * max_params.C);

    var max_val: f32 = -1e30;
    var max_idx: u32 = 0u;
    for (var kh: u32 = 0u; kh < max_params.kH; kh++) {
        for (var kw: u32 = 0u; kw < max_params.kW; kw++) {
            let ih = oh * max_params.kH + kh;
            let iw = ow * max_params.kW + kw;
            if (ih < max_params.H && iw < max_params.W) {
                let pos = n * max_params.C * max_params.H * max_params.W + c * max_params.H * max_params.W + ih * max_params.W + iw;
                if (max_inp[pos] > max_val) {
                    max_val = max_inp[pos];
                    max_idx = pos;
                }
            }
        }
    }
    max_out[idx] = max_val;
    argmax_out[idx] = max_idx;
}
