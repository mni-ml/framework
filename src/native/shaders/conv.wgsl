@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;

struct Conv1dParams { N: u32, C_in: u32, L: u32, C_out: u32, K: u32, stride: u32, padding: u32, L_out: u32 }
@group(0) @binding(3) var<uniform> params: Conv1dParams;

@compute @workgroup_size(256)
fn conv1d_forward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params.N * params.C_out * params.L_out;
    if (idx >= total) { return; }
    let l = idx % params.L_out;
    let co = (idx / params.L_out) % params.C_out;
    let n = idx / (params.L_out * params.C_out);

    var sum_val: f32 = 0.0;
    for (var ci: u32 = 0u; ci < params.C_in; ci++) {
        for (var k: u32 = 0u; k < params.K; k++) {
            let il = i32(l * params.stride) - i32(params.padding) + i32(k);
            if (il >= 0 && u32(il) < params.L) {
                sum_val += inp[n * params.C_in * params.L + ci * params.L + u32(il)]
                         * weight[co * params.C_in * params.K + ci * params.K + k];
            }
        }
    }
    out[idx] = sum_val;
}

@compute @workgroup_size(256)
fn conv1d_backward_input_f32(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params.N * params.C_in * params.L;
    if (idx >= total) { return; }
    let il = idx % params.L;
    let ci = (idx / params.L) % params.C_in;
    let n = idx / (params.L * params.C_in);

    var sum_val: f32 = 0.0;
    for (var co: u32 = 0u; co < params.C_out; co++) {
        for (var k: u32 = 0u; k < params.K; k++) {
            let ol_signed = i32(il) + i32(params.padding) - i32(k);
            if (ol_signed >= 0 && ol_signed % i32(params.stride) == 0) {
                let ol = u32(ol_signed) / params.stride;
                if (ol < params.L_out) {
                    sum_val += out[n * params.C_out * params.L_out + co * params.L_out + ol]
                             * weight[co * params.C_in * params.K + ci * params.K + k];
                }
            }
        }
    }
    inp[idx] = sum_val;  // repurposing inp as dinp output
}

@compute @workgroup_size(256)
fn conv1d_backward_weight_f32(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params.C_out * params.C_in * params.K;
    if (idx >= total) { return; }
    let k = idx % params.K;
    let ci = (idx / params.K) % params.C_in;
    let co = idx / (params.K * params.C_in);

    var sum_val: f32 = 0.0;
    for (var n: u32 = 0u; n < params.N; n++) {
        for (var ol: u32 = 0u; ol < params.L_out; ol++) {
            let il = i32(ol * params.stride) - i32(params.padding) + i32(k);
            if (il >= 0 && u32(il) < params.L) {
                sum_val += out[n * params.C_out * params.L_out + co * params.L_out + ol]
                         * inp[n * params.C_in * params.L + ci * params.L + u32(il)];
            }
        }
    }
    weight[idx] = sum_val;  // repurposing weight as dweight output
}

struct Conv2dParams { N: u32, C_in: u32, H: u32, W: u32, C_out: u32, kH: u32, kW: u32, stride: u32, padding: u32, H_out: u32, W_out: u32 }
@group(1) @binding(0) var<uniform> params2d: Conv2dParams;

@compute @workgroup_size(256)
fn conv2d_forward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params2d.N * params2d.C_out * params2d.H_out * params2d.W_out;
    if (idx >= total) { return; }
    let ow = idx % params2d.W_out;
    let oh = (idx / params2d.W_out) % params2d.H_out;
    let co = (idx / (params2d.W_out * params2d.H_out)) % params2d.C_out;
    let n = idx / (params2d.W_out * params2d.H_out * params2d.C_out);

    var sum_val: f32 = 0.0;
    for (var ci: u32 = 0u; ci < params2d.C_in; ci++) {
        for (var kh: u32 = 0u; kh < params2d.kH; kh++) {
            for (var kw: u32 = 0u; kw < params2d.kW; kw++) {
                let ih = i32(oh * params2d.stride) - i32(params2d.padding) + i32(kh);
                let iw = i32(ow * params2d.stride) - i32(params2d.padding) + i32(kw);
                if (ih >= 0 && u32(ih) < params2d.H && iw >= 0 && u32(iw) < params2d.W) {
                    sum_val += inp[n * params2d.C_in * params2d.H * params2d.W + ci * params2d.H * params2d.W + u32(ih) * params2d.W + u32(iw)]
                             * weight[co * params2d.C_in * params2d.kH * params2d.kW + ci * params2d.kH * params2d.kW + kh * params2d.kW + kw];
                }
            }
        }
    }
    out[idx] = sum_val;
}

@compute @workgroup_size(256)
fn conv2d_backward_input_f32(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params2d.N * params2d.C_in * params2d.H * params2d.W;
    if (idx >= total) { return; }
    let iw = idx % params2d.W;
    let ih = (idx / params2d.W) % params2d.H;
    let ci = (idx / (params2d.W * params2d.H)) % params2d.C_in;
    let n = idx / (params2d.W * params2d.H * params2d.C_in);

    var sum_val: f32 = 0.0;
    for (var co: u32 = 0u; co < params2d.C_out; co++) {
        for (var kh: u32 = 0u; kh < params2d.kH; kh++) {
            for (var kw: u32 = 0u; kw < params2d.kW; kw++) {
                let oh_signed = i32(ih) + i32(params2d.padding) - i32(kh);
                let ow_signed = i32(iw) + i32(params2d.padding) - i32(kw);
                if (oh_signed >= 0 && oh_signed % i32(params2d.stride) == 0 &&
                    ow_signed >= 0 && ow_signed % i32(params2d.stride) == 0) {
                    let oh = u32(oh_signed) / params2d.stride;
                    let ow_val = u32(ow_signed) / params2d.stride;
                    if (oh < params2d.H_out && ow_val < params2d.W_out) {
                        sum_val += out[n * params2d.C_out * params2d.H_out * params2d.W_out + co * params2d.H_out * params2d.W_out + oh * params2d.W_out + ow_val]
                                 * weight[co * params2d.C_in * params2d.kH * params2d.kW + ci * params2d.kH * params2d.kW + kh * params2d.kW + kw];
                    }
                }
            }
        }
    }
    inp[idx] = sum_val;
}

@compute @workgroup_size(256)
fn conv2d_backward_weight_f32(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params2d.C_out * params2d.C_in * params2d.kH * params2d.kW;
    if (idx >= total) { return; }
    let kw = idx % params2d.kW;
    let kh = (idx / params2d.kW) % params2d.kH;
    let ci = (idx / (params2d.kW * params2d.kH)) % params2d.C_in;
    let co = idx / (params2d.kW * params2d.kH * params2d.C_in);

    var sum_val: f32 = 0.0;
    for (var n: u32 = 0u; n < params2d.N; n++) {
        for (var oh: u32 = 0u; oh < params2d.H_out; oh++) {
            for (var ow: u32 = 0u; ow < params2d.W_out; ow++) {
                let ih = i32(oh * params2d.stride) - i32(params2d.padding) + i32(kh);
                let iw = i32(ow * params2d.stride) - i32(params2d.padding) + i32(kw);
                if (ih >= 0 && u32(ih) < params2d.H && iw >= 0 && u32(iw) < params2d.W) {
                    sum_val += out[n * params2d.C_out * params2d.H_out * params2d.W_out + co * params2d.H_out * params2d.W_out + oh * params2d.W_out + ow]
                             * inp[n * params2d.C_in * params2d.H * params2d.W + ci * params2d.H * params2d.W + u32(ih) * params2d.W + u32(iw)];
                }
            }
        }
    }
    weight[idx] = sum_val;
}
