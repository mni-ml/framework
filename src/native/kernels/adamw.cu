extern "C" __global__
void adamw_step_f32(float* param, float* m, float* v, const float* grad,
                    float lr, float beta1, float beta2, float eps,
                    float weight_decay, float bc1, float bc2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;

    float m_hat = mi / bc1;
    float v_hat = vi / bc2;

    param[i] = param[i] * (1.0f - lr * weight_decay) - lr * m_hat / (sqrtf(v_hat) + eps);
}
