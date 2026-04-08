extern "C" __global__
void sum_along_dim_f32(float* out, const float* inp, int outer, int dim_size, int inner, int total_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    int o = idx / inner;
    int j = idx % inner;

    float sum = 0.0f;
    for (int d = 0; d < dim_size; d++) {
        sum += inp[(o * dim_size + d) * inner + j];
    }
    out[idx] = sum;
}

extern "C" __global__
void mean_along_dim_f32(float* out, const float* inp, int outer, int dim_size, int inner, int total_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    int o = idx / inner;
    int j = idx % inner;

    float sum = 0.0f;
    for (int d = 0; d < dim_size; d++) {
        sum += inp[(o * dim_size + d) * inner + j];
    }
    out[idx] = sum / (float)dim_size;
}

extern "C" __global__
void max_along_dim_f32(float* out, const float* inp, int outer, int dim_size, int inner, int total_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    int o = idx / inner;
    int j = idx % inner;

    float mx = inp[o * dim_size * inner + j];
    for (int d = 1; d < dim_size; d++) {
        float val = inp[(o * dim_size + d) * inner + j];
        if (val > mx) mx = val;
    }
    out[idx] = mx;
}

extern "C" __global__
void sum_broadcast_f32(float* out, const float* grad, int outer, int dim_size, int inner, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int tmp = idx / inner;
    int j = idx % inner;
    int o = tmp / dim_size;

    out[idx] = grad[o * inner + j];
}
