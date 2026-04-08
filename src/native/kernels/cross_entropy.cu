extern "C" __global__
void cross_entropy_forward_f32(float* losses, float* softmax_out,
                               const float* logits, const int* targets,
                               int n, int v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* row = logits + i * v;
    float* sm_row = softmax_out + i * v;

    float max_val = row[0];
    for (int j = 1; j < v; j++) {
        if (row[j] > max_val) max_val = row[j];
    }

    float sum = 0.0f;
    for (int j = 0; j < v; j++) {
        float e = expf(row[j] - max_val);
        sm_row[j] = e;
        sum += e;
    }

    float inv_sum = 1.0f / sum;
    for (int j = 0; j < v; j++) {
        sm_row[j] *= inv_sum;
    }

    int t = targets[i];
    losses[i] = -logf(sm_row[t] + 1e-9f);
}

extern "C" __global__
void cross_entropy_backward_f32(float* dlogits, const float* softmax_out,
                                const int* targets, float grad_scale,
                                int n, int v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* sm_row = softmax_out + i * v;
    float* dl_row = dlogits + i * v;
    int t = targets[i];

    for (int j = 0; j < v; j++) {
        float indicator = (j == t) ? 1.0f : 0.0f;
        dl_row[j] = (sm_row[j] - indicator) * grad_scale;
    }
}
