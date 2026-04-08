extern "C" __global__
void sample_batch_i32(int* out_inputs, int* out_targets,
                      const int* dataset, int dataset_len,
                      int block_size, int batch_size,
                      const int* offsets) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    int start = offsets[b];
    int inp_off = b * block_size;
    int tgt_off = b * block_size;

    for (int t = 0; t < block_size; t++) {
        out_inputs[inp_off + t] = dataset[start + t];
        out_targets[tgt_off + t] = dataset[start + t + 1];
    }
}
