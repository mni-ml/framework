extern "C" __global__
void gelu_forward_f32(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float cube = xi * xi * xi;
        float inner = 0.7978845608f * (xi + 0.044715f * cube); // sqrt(2/pi) ≈ 0.7978845608
        out[i] = 0.5f * xi * (1.0f + tanhf(inner));
    }
}

extern "C" __global__
void gelu_backward_f32(float* dx, const float* dy, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float x2 = xi * xi;
        float cube = x2 * xi;
        float inner = 0.7978845608f * (xi + 0.044715f * cube);
        float tanh_val = tanhf(inner);
        float sech2 = 1.0f - tanh_val * tanh_val;
        float d_inner = 0.7978845608f * (1.0f + 0.134145f * x2); // 3 * 0.044715 = 0.134145
        float grad = 0.5f * (1.0f + tanh_val) + 0.5f * xi * sech2 * d_inner;
        dx[i] = dy[i] * grad;
    }
}

extern "C" __global__
void sigmoid_forward_f32(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / (1.0f + expf(-x[i]));
}

extern "C" __global__
void sigmoid_backward_f32(float* dx, const float* dy, const float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = dy[i] * out[i] * (1.0f - out[i]);
}

extern "C" __global__
void relu_forward_f32(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

extern "C" __global__
void relu_backward_f32(float* dx, const float* dy, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
}
