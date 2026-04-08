// bf16 conversion kernels for mixed-precision training.
// bf16 = upper 16 bits of f32 (1 sign + 8 exponent + 7 mantissa).

extern "C" __global__
void f32_to_bf16(unsigned short* out, const float* in_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned int f = __float_as_uint(in_data[i]);
    // Round-to-nearest-even: add bias + check LSB of bf16 result
    unsigned int rounding = (f + 0x7FFFu + ((f >> 16) & 1u)) >> 16;
    out[i] = (unsigned short)rounding;
}

extern "C" __global__
void bf16_to_f32(float* out, const unsigned short* in_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned int f = ((unsigned int)in_data[i]) << 16;
    out[i] = __uint_as_float(f);
}

// Scale all gradients by a factor (for GradScaler unscale step)
extern "C" __global__
void scale_f32(float* data, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] *= scale;
}

// Check for inf/nan in gradient data (returns 1.0 if found, 0.0 if clean)
extern "C" __global__
void check_inf_nan_f32(float* result, const float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (isinf(data[i]) || isnan(data[i])) {
        result[0] = 1.0f;
    }
}
