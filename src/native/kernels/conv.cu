// Conv1d forward: input [N,C_in,L], weight [C_out,C_in,K] -> output [N,C_out,L_out]
extern "C" __global__
void conv1d_forward_f32(
    float* out, const float* inp, const float* weight,
    int N, int C_in, int L, int C_out, int K, int stride, int padding, int L_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (idx >= total) return;

    int l = idx % L_out;
    int co = (idx / L_out) % C_out;
    int n = idx / (L_out * C_out);

    float sum = 0.0f;
    for (int ci = 0; ci < C_in; ci++) {
        for (int k = 0; k < K; k++) {
            int il = l * stride - padding + k;
            if (il >= 0 && il < L) {
                sum += inp[n * C_in * L + ci * L + il] * weight[co * C_in * K + ci * K + k];
            }
        }
    }
    out[idx] = sum;
}

// Conv1d backward w.r.t. input
extern "C" __global__
void conv1d_backward_input_f32(
    float* dinp, const float* dout, const float* weight,
    int N, int C_in, int L, int C_out, int K, int stride, int padding, int L_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * L;
    if (idx >= total) return;

    int il = idx % L;
    int ci = (idx / L) % C_in;
    int n = idx / (L * C_in);

    float sum = 0.0f;
    for (int co = 0; co < C_out; co++) {
        for (int k = 0; k < K; k++) {
            int ol = il + padding - k;
            if (ol >= 0 && ol % stride == 0) {
                ol /= stride;
                if (ol < L_out) {
                    sum += dout[n * C_out * L_out + co * L_out + ol] * weight[co * C_in * K + ci * K + k];
                }
            }
        }
    }
    dinp[idx] = sum;
}

// Conv1d backward w.r.t. weight
extern "C" __global__
void conv1d_backward_weight_f32(
    float* dweight, const float* dout, const float* inp,
    int N, int C_in, int L, int C_out, int K, int stride, int padding, int L_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * C_in * K;
    if (idx >= total) return;

    int k = idx % K;
    int ci = (idx / K) % C_in;
    int co = idx / (K * C_in);

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int ol = 0; ol < L_out; ol++) {
            int il = ol * stride - padding + k;
            if (il >= 0 && il < L) {
                sum += dout[n * C_out * L_out + co * L_out + ol] * inp[n * C_in * L + ci * L + il];
            }
        }
    }
    dweight[idx] = sum;
}

// Conv2d forward: input [N,C_in,H,W], weight [C_out,C_in,kH,kW] -> output [N,C_out,H_out,W_out]
extern "C" __global__
void conv2d_forward_f32(
    float* out, const float* inp, const float* weight,
    int N, int C_in, int H, int W, int C_out, int kH, int kW,
    int stride, int padding, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int ow = idx % W_out;
    int oh = (idx / W_out) % H_out;
    int co = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);

    float sum = 0.0f;
    for (int ci = 0; ci < C_in; ci++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    sum += inp[n * C_in * H * W + ci * H * W + ih * W + iw]
                         * weight[co * C_in * kH * kW + ci * kH * kW + kh * kW + kw];
                }
            }
        }
    }
    out[idx] = sum;
}

// Conv2d backward w.r.t. input
extern "C" __global__
void conv2d_backward_input_f32(
    float* dinp, const float* dout, const float* weight,
    int N, int C_in, int H, int W, int C_out, int kH, int kW,
    int stride, int padding, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * H * W;
    if (idx >= total) return;

    int iw = idx % W;
    int ih = (idx / W) % H;
    int ci = (idx / (W * H)) % C_in;
    int n = idx / (W * H * C_in);

    float sum = 0.0f;
    for (int co = 0; co < C_out; co++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int oh = ih + padding - kh;
                int ow = iw + padding - kw;
                if (oh >= 0 && oh % stride == 0 && ow >= 0 && ow % stride == 0) {
                    oh /= stride;
                    ow /= stride;
                    if (oh < H_out && ow < W_out) {
                        sum += dout[n * C_out * H_out * W_out + co * H_out * W_out + oh * W_out + ow]
                             * weight[co * C_in * kH * kW + ci * kH * kW + kh * kW + kw];
                    }
                }
            }
        }
    }
    dinp[idx] = sum;
}

// Conv2d backward w.r.t. weight
extern "C" __global__
void conv2d_backward_weight_f32(
    float* dweight, const float* dout, const float* inp,
    int N, int C_in, int H, int W, int C_out, int kH, int kW,
    int stride, int padding, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * C_in * kH * kW;
    if (idx >= total) return;

    int kw = idx % kW;
    int kh = (idx / kW) % kH;
    int ci = (idx / (kW * kH)) % C_in;
    int co = idx / (kW * kH * C_in);

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    sum += dout[n * C_out * H_out * W_out + co * H_out * W_out + oh * W_out + ow]
                         * inp[n * C_in * H * W + ci * H * W + ih * W + iw];
                }
            }
        }
    }
    dweight[idx] = sum;
}
