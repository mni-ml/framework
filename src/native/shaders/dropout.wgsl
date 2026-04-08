@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;

struct Params { n: u32, scale: f32 }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn dropout_apply_f32(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.n) {
        out[i] = x[i] * mask[i] * params.scale;
    }
}
