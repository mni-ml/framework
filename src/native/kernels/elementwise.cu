extern "C" __global__
void add_f32(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" __global__
void sub_f32(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

extern "C" __global__
void mul_f32(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

extern "C" __global__
void neg_f32(float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -a[i];
}

extern "C" __global__
void mul_scalar_f32(float* out, const float* a, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * s;
}

extern "C" __global__
void exp_f32(float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(a[i]);
}

extern "C" __global__
void log_f32(float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = logf(a[i]);
}

extern "C" __global__
void sin_f32(float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sinf(a[i]);
}

extern "C" __global__
void sin_backward_f32(float* dx, const float* dy, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = dy[i] * cosf(x[i]);
}

extern "C" __global__
void cos_f32(float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cosf(a[i]);
}

extern "C" __global__
void cos_backward_f32(float* dx, const float* dy, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = -dy[i] * sinf(x[i]);
}

extern "C" __global__
void sqrt_f32(float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sqrtf(fmaxf(a[i], 0.0f));
}

extern "C" __global__
void sqrt_backward_f32(float* dx, const float* dy, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = x[i] > 0.0f ? dy[i] * 0.5f / sqrtf(x[i]) : 0.0f;
}

extern "C" __global__
void add_bias_f32(float* out, const float* a, const float* bias, int total, int bias_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = a[i] + bias[i % bias_size];
}

extern "C" __global__
void fill_f32(float* out, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = val;
}

extern "C" __global__
void div_f32(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / b[i];
}

extern "C" __global__
void copy_f32(float* out, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = src[i];
}

extern "C" __global__
void permute_f32(float* out, const float* src, int n,
                 int ds0, int ds1, int ds2, int ds3,
                 int es0, int es1, int es2, int es3,
                 int ndim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int rem = idx;
    int src_idx = 0;
    if (ndim >= 1) { int c = rem / ds0; rem %= ds0; src_idx += c * es0; }
    if (ndim >= 2) { int c = rem / ds1; rem %= ds1; src_idx += c * es1; }
    if (ndim >= 3) { int c = rem / ds2; rem %= ds2; src_idx += c * es2; }
    if (ndim >= 4) { src_idx += rem * es3; }
    out[idx] = src[src_idx];
}

extern "C" __global__
void broadcast_add_f32(float* out, const float* a, const float* b,
                       int n, int b_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i % b_size];
}

extern "C" __global__
void broadcast_mul_f32(float* out, const float* a, const float* b,
                       int n, int b_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i % b_size];
}

extern "C" __global__
void lt_f32(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] < b[i] ? 1.0f : 0.0f;
}

extern "C" __global__
void eq_f32(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (fabsf(a[i] - b[i]) < 1e-6f) ? 1.0f : 0.0f;
}

extern "C" __global__
void gt_f32(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] > b[i] ? 1.0f : 0.0f;
}

extern "C" __global__
void is_close_f32(float* out, const float* a, const float* b, float tol, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (fabsf(a[i] - b[i]) < tol) ? 1.0f : 0.0f;
}

extern "C" __global__
void pow_f32(float* out, const float* a, float exponent, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = powf(a[i], exponent);
}

extern "C" __global__
void pow_backward_f32(float* dx, const float* dy, const float* x, float exponent, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = dy[i] * exponent * powf(x[i], exponent - 1.0f);
}

extern "C" __global__
void div_backward_a_f32(float* da, const float* dy, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) da[i] = dy[i] / b[i];
}

extern "C" __global__
void div_backward_b_f32(float* db, const float* dy, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) db[i] = -dy[i] * a[i] / (b[i] * b[i]);
}

extern "C" __global__
void sum_reduce_all_f32(float* out, const float* inp, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? inp[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}
