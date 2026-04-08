@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;

struct Params { batch: u32, seq_len: u32, emb_dim: u32 }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn embedding_forward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params.batch * params.seq_len * params.emb_dim;
    if (idx >= total) { return; }
    let d = idx % params.emb_dim;
    let s = (idx / params.emb_dim) % params.seq_len;
    let b = idx / (params.emb_dim * params.seq_len);
    let token_id = indices[b * params.seq_len + s];
    out[idx] = weight[token_id * params.emb_dim + d];
}

@group(0) @binding(0) var<storage, read_write> dweight: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> dy_emb: array<f32>;
@group(0) @binding(2) var<storage, read> indices_bwd: array<u32>;

struct ParamsBwd { batch: u32, seq_len: u32, emb_dim: u32 }
@group(0) @binding(3) var<uniform> params_bwd: ParamsBwd;

@compute @workgroup_size(256)
fn embedding_backward_f32(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params_bwd.batch * params_bwd.seq_len * params_bwd.emb_dim;
    if (idx >= total) { return; }
    let d = idx % params_bwd.emb_dim;
    let s = (idx / params_bwd.emb_dim) % params_bwd.seq_len;
    let b = idx / (params_bwd.emb_dim * params_bwd.seq_len);
    let token_id = indices_bwd[b * params_bwd.seq_len + s];
    let val = dy_emb[idx];
    let target = token_id * params_bwd.emb_dim + d;
    // WebGPU doesn't have f32 atomicAdd, use bitcast workaround
    let bits = bitcast<u32>(val);
    atomicAdd(&dweight[target], bits);
}
