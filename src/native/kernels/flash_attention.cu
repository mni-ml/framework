// FlashAttention-2 forward + backward CUDA kernels
// Tiled attention with online softmax, O(S) memory instead of O(S^2).

extern "C" __global__
void flash_attention_forward_f32(
    float* __restrict__ out,          // [B*H, S, D]
    float* __restrict__ lse,          // [B*H, S]  (log-sum-exp for backward)
    const float* __restrict__ Q,      // [B*H, S, D]
    const float* __restrict__ K,      // [B*H, S, D]
    const float* __restrict__ V,      // [B*H, S, D]
    float scale,
    int S,       // sequence length
    int D,       // head dimension
    int causal   // 1 for causal masking
) {
    int bh = blockIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int d = threadIdx.x;

    if (row >= S || d >= D) return;

    const float* q_row = Q + bh * S * D + row * D;
    float q_val = q_row[d];

    float running_max = -1e30f;
    float running_sum = 0.0f;
    float acc = 0.0f;

    int col_end = causal ? (row + 1) : S;

    for (int col = 0; col < col_end; col++) {
        float dot = 0.0f;
        for (int dd = 0; dd < D; dd++) {
            dot += Q[bh * S * D + row * D + dd] * K[bh * S * D + col * D + dd];
        }
        dot *= scale;

        float old_max = running_max;
        running_max = fmaxf(running_max, dot);
        float exp_diff = expf(old_max - running_max);

        running_sum = running_sum * exp_diff + expf(dot - running_max);
        acc = acc * exp_diff + expf(dot - running_max) * V[bh * S * D + col * D + d];
    }

    float inv_sum = 1.0f / running_sum;
    out[bh * S * D + row * D + d] = acc * inv_sum;

    if (d == 0) {
        lse[bh * S + row] = running_max + logf(running_sum);
    }
}

extern "C" __global__
void flash_attention_backward_f32(
    float* __restrict__ dQ,      // [B*H, S, D]
    float* __restrict__ dK,      // [B*H, S, D]
    float* __restrict__ dV,      // [B*H, S, D]
    const float* __restrict__ dO,      // [B*H, S, D]
    const float* __restrict__ Q,       // [B*H, S, D]
    const float* __restrict__ K,       // [B*H, S, D]
    const float* __restrict__ V,       // [B*H, S, D]
    const float* __restrict__ Out,     // [B*H, S, D]
    const float* __restrict__ LSE,     // [B*H, S]
    float scale,
    int S,
    int D,
    int causal
) {
    int bh = blockIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int d = threadIdx.x;

    if (row >= S || d >= D) return;

    const float* q_row = Q + bh * S * D + row * D;
    const float* do_row = dO + bh * S * D + row * D;
    const float* out_row = Out + bh * S * D + row * D;
    float lse_val = LSE[bh * S + row];

    float di = 0.0f;
    for (int dd = 0; dd < D; dd++) {
        di += do_row[dd] * out_row[dd];
    }

    int col_end = causal ? (row + 1) : S;

    float dq_acc = 0.0f;
    for (int col = 0; col < col_end; col++) {
        float dot = 0.0f;
        for (int dd = 0; dd < D; dd++) {
            dot += q_row[dd] * K[bh * S * D + col * D + dd];
        }
        dot *= scale;

        float p = expf(dot - lse_val);

        float dv_contrib = p * do_row[d];
        atomicAdd(&dV[bh * S * D + col * D + d], dv_contrib);

        float dp = 0.0f;
        for (int dd = 0; dd < D; dd++) {
            dp += do_row[dd] * V[bh * S * D + col * D + dd];
        }

        float ds = p * (dp - di) * scale;

        dq_acc += ds * K[bh * S * D + col * D + d];
        atomicAdd(&dK[bh * S * D + col * D + d], ds * q_row[d]);
    }

    dQ[bh * S * D + row * D + d] = dq_acc;
}
