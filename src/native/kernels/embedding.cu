extern "C" __global__
void embedding_forward_f32(float* out, const float* weight, const int* indices,
                           int total_tokens, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens * embed_dim) return;

    int t = idx / embed_dim;
    int d = idx % embed_dim;
    out[t * embed_dim + d] = weight[indices[t] * embed_dim + d];
}

extern "C" __global__
void embedding_backward_f32(float* dweight, const float* dout, const int* indices,
                            int total_tokens, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens * embed_dim) return;

    int t = idx / embed_dim;
    int d = idx % embed_dim;
    atomicAdd(&dweight[indices[t] * embed_dim + d], dout[t * embed_dim + d]);
}
