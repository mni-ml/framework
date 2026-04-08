#define BLOCK_DIM 256
#define NEG_INF __int_as_float(0xff800000)

extern "C" __global__
void softmax_forward_f32(float* out, const float* x, int outer, int dim_size, int inner) {
    int row = blockIdx.x;
    int total = outer * inner;
    if (row >= total) return;

    int tid = threadIdx.x;
    int o = row / inner;
    int j = row % inner;

    __shared__ float sdata[BLOCK_DIM];

    float local_max = NEG_INF;
    for (int d = tid; d < dim_size; d += blockDim.x) {
        float v = x[(o * dim_size + d) * inner + j];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int d = tid; d < dim_size; d += blockDim.x) {
        float e = expf(x[(o * dim_size + d) * inner + j] - max_val);
        out[(o * dim_size + d) * inner + j] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    for (int d = tid; d < dim_size; d += blockDim.x)
        out[(o * dim_size + d) * inner + j] *= inv_sum;
}

extern "C" __global__
void softmax_backward_f32(float* dx, const float* dy, const float* out, int outer, int dim_size, int inner) {
    int row = blockIdx.x;
    int total = outer * inner;
    if (row >= total) return;

    int tid = threadIdx.x;
    int o = row / inner;
    int j = row % inner;

    __shared__ float sdata[BLOCK_DIM];

    float local_dot = 0.0f;
    for (int d = tid; d < dim_size; d += blockDim.x) {
        int pos = (o * dim_size + d) * inner + j;
        local_dot += dy[pos] * out[pos];
    }
    sdata[tid] = local_dot;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float dot = sdata[0];

    for (int d = tid; d < dim_size; d += blockDim.x) {
        int pos = (o * dim_size + d) * inner + j;
        dx[pos] = out[pos] * (dy[pos] - dot);
    }
}
