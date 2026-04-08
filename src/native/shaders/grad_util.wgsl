@group(0) @binding(0) var<storage, read_write> partial: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;

struct Params { n: u32, num_blocks: u32 }
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn grad_norm_sq_partial_f32(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
    let i = gid.x;
    let tid = lid.x;
    if (i < params.n) {
        sdata[tid] = grad[i] * grad[i];
    } else {
        sdata[tid] = 0.0;
    }
    workgroupBarrier();
    var s: u32 = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { sdata[tid] += sdata[tid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (tid == 0u) { partial[wid.x] = sdata[0]; }
}

@group(0) @binding(0) var<storage, read_write> grad_clip: array<f32>;

struct ClipParams { n: u32, clip_coeff: f32 }
@group(0) @binding(1) var<uniform> clip_params: ClipParams;

@compute @workgroup_size(256)
fn grad_clip_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < clip_params.n) {
        grad_clip[i] = grad_clip[i] * clip_params.clip_coeff;
    }
}
