#define BLOCK_DIM 256

// Fused residual + layernorm: out = layernorm(x + residual, gamma, beta, eps)
// Saves a kernel launch and a global memory round-trip vs separate add + layernorm.
extern "C" __global__
void residual_layernorm_forward_f32(
    float* out,
    float* residual_out,   // x + residual stored for backward
    float* mean_out,
    float* rstd_out,
    const float* x,
    const float* residual,
    const float* gamma,
    const float* beta,
    int n, int c, float eps
) {
    int row = blockIdx.x;
    if (row >= n) return;
    int tid = threadIdx.x;

    const float* row_x = x + row * c;
    const float* row_r = residual + row * c;
    float* row_res = residual_out + row * c;
    float* row_out = out + row * c;

    __shared__ float sdata[BLOCK_DIM];

    float local_sum = 0.0f;
    for (int j = tid; j < c; j += blockDim.x) {
        float val = row_x[j] + row_r[j];
        row_res[j] = val;
        local_sum += val;
    }
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
        float diff = row_res[j] - mean;
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
        row_out[j] = gamma[j] * (row_res[j] - mean) * rstd + beta[j];

    if (tid == 0) {
        if (mean_out) mean_out[row] = mean;
        if (rstd_out) rstd_out[row] = rstd;
    }
}

// Fused bias + GELU: out = gelu(x + bias)
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
extern "C" __global__
void bias_gelu_forward_f32(
    float* out, const float* x, const float* bias,
    int n, int c
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * c) return;

    int j = idx % c;
    float val = x[idx] + bias[j];

    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    const float COEFF = 0.044715f;
    float inner = SQRT_2_OVER_PI * (val + COEFF * val * val * val);
    out[idx] = 0.5f * val * (1.0f + tanhf(inner));
}

// Backward for bias+gelu
extern "C" __global__
void bias_gelu_backward_f32(
    float* dx, float* dbias_partial,
    const float* grad, const float* x, const float* bias,
    int n, int c
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * c) return;

    int j = idx % c;
    float val = x[idx] + bias[j];

    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    const float COEFF = 0.044715f;
    float inner = SQRT_2_OVER_PI * (val + COEFF * val * val * val);
    float th = tanhf(inner);
    float sech2 = 1.0f - th * th;
    float d_inner = SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * val * val);
    float dgelu = 0.5f * (1.0f + th) + 0.5f * val * sech2 * d_inner;

    float g = grad[idx] * dgelu;
    dx[idx] = g;
    atomicAdd(&dbias_partial[j], g);
}
