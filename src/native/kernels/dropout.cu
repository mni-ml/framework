extern "C" __global__
void dropout_apply_f32(float* out, const float* x, const float* mask, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * mask[i] * scale;
}

extern "C" __global__
void dropout_backward_f32(float* dx, const float* dy, const float* mask, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = dy[i] * mask[i] * scale;
}
