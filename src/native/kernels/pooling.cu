// Average pooling 2D: input [N,C,H,W] -> output [N,C,H_out,W_out]
extern "C" __global__
void avgpool2d_forward_f32(
    float* out, const float* inp,
    int N, int C, int H, int W, int kH, int kW, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int ow = idx % W_out;
    int oh = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);

    float sum = 0.0f;
    int count = 0;
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int ih = oh * kH + kh;
            int iw = ow * kW + kw;
            if (ih < H && iw < W) {
                sum += inp[n * C * H * W + c * H * W + ih * W + iw];
                count++;
            }
        }
    }
    out[idx] = sum / (float)count;
}

// Average pooling 2D backward
extern "C" __global__
void avgpool2d_backward_f32(
    float* dinp, const float* dout,
    int N, int C, int H, int W, int kH, int kW, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int iw = idx % W;
    int ih = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    int oh = ih / kH;
    int ow = iw / kW;
    if (oh < H_out && ow < W_out) {
        float inv = 1.0f / (float)(kH * kW);
        dinp[idx] = dout[n * C * H_out * W_out + c * H_out * W_out + oh * W_out + ow] * inv;
    } else {
        dinp[idx] = 0.0f;
    }
}

// Max pooling 2D forward (saves argmax for backward)
extern "C" __global__
void maxpool2d_forward_f32(
    float* out, int* argmax, const float* inp,
    int N, int C, int H, int W, int kH, int kW, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int ow = idx % W_out;
    int oh = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);

    float max_val = -1e30f;
    int max_idx = 0;
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int ih = oh * kH + kh;
            int iw = ow * kW + kw;
            if (ih < H && iw < W) {
                int pos = n * C * H * W + c * H * W + ih * W + iw;
                if (inp[pos] > max_val) {
                    max_val = inp[pos];
                    max_idx = pos;
                }
            }
        }
    }
    out[idx] = max_val;
    argmax[idx] = max_idx;
}

// Max pooling 2D backward (scatter gradient to argmax positions)
extern "C" __global__
void maxpool2d_backward_f32(
    float* dinp, const float* dout, const int* argmax,
    int out_size, int inp_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;
    atomicAdd(&dinp[argmax[idx]], dout[idx]);
}
