#define BLOCK_DIM 256

extern "C" __global__
void layernorm_forward_f32(float* out, float* mean_out, float* rstd_out,
                           const float* x, const float* gamma, const float* beta,
                           int n, int c, float eps) {
    int row = blockIdx.x;
    if (row >= n) return;

    int tid = threadIdx.x;
    const float* row_x = x + row * c;
    float* row_out = out + row * c;

    __shared__ float sdata[BLOCK_DIM];

    float local_sum = 0.0f;
    for (int j = tid; j < c; j += blockDim.x)
        local_sum += row_x[j];
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)c;
    __syncthreads();

    float local_var = 0.0f;
    for (int j = tid; j < c; j += blockDim.x) {
        float diff = row_x[j] - mean;
        local_var += diff * diff;
    }
    sdata[tid] = local_var;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rstd = rsqrtf(sdata[0] / (float)c + eps);

    for (int j = tid; j < c; j += blockDim.x)
        row_out[j] = gamma[j] * (row_x[j] - mean) * rstd + beta[j];

    if (tid == 0) {
        if (mean_out) mean_out[row] = mean;
        if (rstd_out) rstd_out[row] = rstd;
    }
}

extern "C" __global__
void layernorm_backward_f32(float* dx, float* dgamma, float* dbeta,
                            const float* dy, const float* x,
                            const float* mean, const float* rstd, const float* gamma,
                            int n, int c) {
    int row = blockIdx.x;
    if (row >= n) return;

    int tid = threadIdx.x;
    const float* row_dy = dy + row * c;
    const float* row_x = x + row * c;
    float* row_dx = dx + row * c;
    float m = mean[row];
    float r = rstd[row];

    __shared__ float s_dot_dy_xhat[BLOCK_DIM];
    __shared__ float s_dot_dy[BLOCK_DIM];

    float local_dot_dy_xhat = 0.0f;
    float local_dot_dy = 0.0f;
    for (int j = tid; j < c; j += blockDim.x) {
        float xhat = (row_x[j] - m) * r;
        local_dot_dy_xhat += row_dy[j] * gamma[j] * xhat;
        local_dot_dy += row_dy[j] * gamma[j];
    }
    s_dot_dy_xhat[tid] = local_dot_dy_xhat;
    s_dot_dy[tid] = local_dot_dy;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_dot_dy_xhat[tid] += s_dot_dy_xhat[tid + s];
            s_dot_dy[tid] += s_dot_dy[tid + s];
        }
        __syncthreads();
    }

    float dot_dy_xhat = s_dot_dy_xhat[0];
    float dot_dy = s_dot_dy[0];
    float inv_c = 1.0f / (float)c;

    for (int j = tid; j < c; j += blockDim.x) {
        float xhat = (row_x[j] - m) * r;
        row_dx[j] = r * (row_dy[j] * gamma[j] - inv_c * (dot_dy + xhat * dot_dy_xhat));
    }

    for (int j = tid; j < c; j += blockDim.x) {
        float xhat = (row_x[j] - m) * r;
        atomicAdd(&dgamma[j], row_dy[j] * xhat);
        atomicAdd(&dbeta[j], row_dy[j]);
    }
}
