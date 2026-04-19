use crate::tensor::{shape_size, TensorId, TensorStore};

#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

const EPS_SCALE: f32 = 1e-8;

#[cfg(feature = "cuda")]
fn launch_cfg(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[derive(Clone, Debug)]
pub struct KvCacheConfig {
    pub batch_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub quantized: bool,
}

impl KvCacheConfig {
    pub fn batch_heads(&self) -> usize {
        self.batch_size * self.num_heads
    }

    fn capacity_elements(&self) -> usize {
        self.batch_heads() * self.max_seq_len * self.head_dim
    }
}

pub struct KvCache {
    cfg: KvCacheConfig,
    len: usize,
    keys_fp32: Vec<f32>,
    vals_fp32: Vec<f32>,
    keys_i8: Vec<i8>,
    vals_i8: Vec<i8>,
    key_scales: Vec<f32>, // [BH, max_seq]
    val_scales: Vec<f32>, // [BH, max_seq]
}

impl KvCache {
    pub fn new(cfg: KvCacheConfig) -> Self {
        let cap = cfg.capacity_elements();
        let scale_cap = cfg.batch_heads() * cfg.max_seq_len;
        Self {
            cfg,
            len: 0,
            keys_fp32: vec![0.0; cap],
            vals_fp32: vec![0.0; cap],
            keys_i8: vec![0; cap],
            vals_i8: vec![0; cap],
            key_scales: vec![1.0; scale_cap],
            val_scales: vec![1.0; scale_cap],
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn quantized(&self) -> bool {
        self.cfg.quantized
    }

    pub fn reset(&mut self) {
        self.len = 0;
        self.keys_fp32.fill(0.0);
        self.vals_fp32.fill(0.0);
        self.keys_i8.fill(0);
        self.vals_i8.fill(0);
        self.key_scales.fill(1.0);
        self.val_scales.fill(1.0);
    }

    fn validate_single_step_shape(
        &self,
        shape: &[usize],
        op_name: &str,
    ) -> Result<(usize, usize, usize, usize), String> {
        let (batch_size, num_heads, seq_len, head_dim) = parse_bhsd(shape)?;
        if seq_len != 1 {
            return Err(format!(
                "kv cache {} expects seq_len=1, got {}",
                op_name, seq_len
            ));
        }
        if batch_size != self.cfg.batch_size
            || num_heads != self.cfg.num_heads
            || head_dim != self.cfg.head_dim
        {
            return Err(format!(
                "kv cache {} shape mismatch: expected [B={}, H={}, S=1, D={}], got [B={}, H={}, S=1, D={}]",
                op_name,
                self.cfg.batch_size,
                self.cfg.num_heads,
                self.cfg.head_dim,
                batch_size,
                num_heads,
                head_dim
            ));
        }
        Ok((batch_size, num_heads, seq_len, head_dim))
    }

    fn append_inner(
        &mut self,
        k: TensorId,
        v: TensorId,
        bh: usize,
        d: usize,
        store: &mut TensorStore,
    ) -> Result<(), String> {
        if self.len >= self.cfg.max_seq_len {
            return Err(format!(
                "kv cache is full (len={}, max_seq_len={})",
                self.len, self.cfg.max_seq_len
            ));
        }
        let pos = self.len;
        if self.cfg.quantized {
            #[cfg(feature = "cuda")]
            {
                let (k_quant, k_scales) = quantize_rows_cuda(k, bh, d, store)?;
                let (v_quant, v_scales) = quantize_rows_cuda(v, bh, d, store)?;
                for row in 0..bh {
                    let cache_scale_idx = row * self.cfg.max_seq_len + pos;
                    self.key_scales[cache_scale_idx] = k_scales[row];
                    self.val_scales[cache_scale_idx] = v_scales[row];

                    let src = row * d;
                    let dst = ((row * self.cfg.max_seq_len) + pos) * d;
                    self.keys_i8[dst..dst + d].copy_from_slice(&k_quant[src..src + d]);
                    self.vals_i8[dst..dst + d].copy_from_slice(&v_quant[src..src + d]);
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let k_data = flatten_single_step(store.to_host(k), bh, d)?;
                let v_data = flatten_single_step(store.to_host(v), bh, d)?;
                for row in 0..bh {
                    let src = row * d;
                    let dst = ((row * self.cfg.max_seq_len) + pos) * d;
                    let (kq, ks) = quantize_row_i8(&k_data[src..src + d]);
                    let (vq, vs) = quantize_row_i8(&v_data[src..src + d]);
                    self.keys_i8[dst..dst + d].copy_from_slice(&kq);
                    self.vals_i8[dst..dst + d].copy_from_slice(&vq);
                    self.key_scales[row * self.cfg.max_seq_len + pos] = ks;
                    self.val_scales[row * self.cfg.max_seq_len + pos] = vs;
                }
            }
        } else {
            let k_data = flatten_single_step(store.to_host(k), bh, d)?;
            let v_data = flatten_single_step(store.to_host(v), bh, d)?;
            for row in 0..bh {
                let src = row * d;
                let dst = ((row * self.cfg.max_seq_len) + pos) * d;
                self.keys_fp32[dst..dst + d].copy_from_slice(&k_data[src..src + d]);
                self.vals_fp32[dst..dst + d].copy_from_slice(&v_data[src..src + d]);
            }
        }
        self.len += 1;
        Ok(())
    }

    pub fn append(&mut self, k: TensorId, v: TensorId, store: &mut TensorStore) -> Result<(), String> {
        let k_shape = store.shape(k).to_vec();
        let v_shape = store.shape(v).to_vec();
        if k_shape != v_shape {
            return Err("k/v shapes must match for kv cache append".to_string());
        }
        let (_, _, _, head_dim) = self.validate_single_step_shape(&k_shape, "append")?;
        self.append_inner(k, v, self.cfg.batch_heads(), head_dim, store)
    }

    pub fn append_and_decode(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        scale: f32,
        store: &mut TensorStore,
    ) -> Result<TensorId, String> {
        let q_shape = store.shape(q).to_vec();
        let k_shape = store.shape(k).to_vec();
        let v_shape = store.shape(v).to_vec();
        if q_shape != k_shape || q_shape != v_shape {
            return Err("q/k/v shapes must match for kv cache decode".to_string());
        }

        let (_, _, _, head_dim) = self.validate_single_step_shape(&q_shape, "decode")?;
        let bh = self.cfg.batch_heads();
        let d = head_dim;
        self.append_inner(k, v, bh, d, store)?;
        let cur_len = self.len;

        let q_data = flatten_single_step(store.to_host(q), bh, d)?;
        let (k_read, v_read) = if self.cfg.quantized {
            #[cfg(feature = "cuda")]
            {
                let (kq, ks) = gather_quantized_rows(
                    &self.keys_i8,
                    &self.key_scales,
                    bh,
                    cur_len,
                    d,
                    self.cfg.max_seq_len,
                );
                let (vq, vs) = gather_quantized_rows(
                    &self.vals_i8,
                    &self.val_scales,
                    bh,
                    cur_len,
                    d,
                    self.cfg.max_seq_len,
                );
                (
                    dequantize_rows_cuda(&kq, &ks, bh * cur_len, d)?,
                    dequantize_rows_cuda(&vq, &vs, bh * cur_len, d)?,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                let (kq, ks) = gather_quantized_rows(
                    &self.keys_i8,
                    &self.key_scales,
                    bh,
                    cur_len,
                    d,
                    self.cfg.max_seq_len,
                );
                let (vq, vs) = gather_quantized_rows(
                    &self.vals_i8,
                    &self.val_scales,
                    bh,
                    cur_len,
                    d,
                    self.cfg.max_seq_len,
                );
                (
                    dequantize_rows_cpu(&kq, &ks, bh * cur_len, d),
                    dequantize_rows_cpu(&vq, &vs, bh * cur_len, d),
                )
            }
        } else {
            (
                gather_fp32_rows(&self.keys_fp32, bh, cur_len, d, self.cfg.max_seq_len),
                gather_fp32_rows(&self.vals_fp32, bh, cur_len, d, self.cfg.max_seq_len),
            )
        };

        let mut out = vec![0.0f32; bh * d];
        for row in 0..bh {
            let q_off = row * d;
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;
            let mut acc = vec![0.0f32; d];
            for col in 0..cur_len {
                let kv_off = (row * cur_len + col) * d;
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q_data[q_off + dd] * k_read[kv_off + dd];
                }
                dot *= scale;
                let old_max = running_max;
                running_max = running_max.max(dot);
                let exp_diff = (old_max - running_max).exp();
                let w = (dot - running_max).exp();
                running_sum = running_sum * exp_diff + w;
                for dd in 0..d {
                    acc[dd] = acc[dd] * exp_diff + w * v_read[kv_off + dd];
                }
            }
            let inv_sum = if running_sum > 0.0 { 1.0 / running_sum } else { 0.0 };
            for dd in 0..d {
                out[q_off + dd] = acc[dd] * inv_sum;
            }
        }

        let out_shape = if q_shape.len() == 4 {
            vec![self.cfg.batch_size, self.cfg.num_heads, 1, self.cfg.head_dim]
        } else {
            vec![bh, 1, d]
        };
        Ok(store.from_vec(out, &out_shape))
    }
}

fn parse_bhsd(shape: &[usize]) -> Result<(usize, usize, usize, usize), String> {
    match shape {
        [b, h, s, d] => Ok((*b, *h, *s, *d)),
        [bh, s, d] => Ok((1, *bh, *s, *d)),
        _ => Err(format!(
            "expected [B,H,S,D] or [BH,S,D], got shape {:?}",
            shape
        )),
    }
}

fn flatten_single_step(data: Vec<f32>, bh: usize, d: usize) -> Result<Vec<f32>, String> {
    if data.len() != bh * d {
        return Err(format!(
            "expected {} elements for single-step tensor, got {}",
            bh * d,
            data.len()
        ));
    }
    Ok(data)
}

fn gather_fp32_rows(
    src: &[f32],
    bh: usize,
    len: usize,
    d: usize,
    max_seq: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; bh * len * d];
    for row in 0..bh {
        for t in 0..len {
            let src_off = ((row * max_seq) + t) * d;
            let dst_off = (row * len + t) * d;
            out[dst_off..dst_off + d].copy_from_slice(&src[src_off..src_off + d]);
        }
    }
    out
}

fn gather_quantized_rows(
    src_i8: &[i8],
    src_scales: &[f32],
    bh: usize,
    len: usize,
    d: usize,
    max_seq: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; bh * len * d];
    let mut scales = vec![1.0f32; bh * len];
    for row in 0..bh {
        for t in 0..len {
            let src_off = ((row * max_seq) + t) * d;
            let dst_off = (row * len + t) * d;
            q[dst_off..dst_off + d].copy_from_slice(&src_i8[src_off..src_off + d]);
            scales[row * len + t] = src_scales[row * max_seq + t];
        }
    }
    (q, scales)
}

fn quantize_row_i8(input: &[f32]) -> (Vec<i8>, f32) {
    let mut max_abs = 0.0f32;
    for &v in input {
        max_abs = max_abs.max(v.abs());
    }
    let scale = (max_abs / 127.0).max(EPS_SCALE);
    let inv = 1.0 / scale;
    let mut out = vec![0i8; input.len()];
    for i in 0..input.len() {
        let q = (input[i] * inv).round().clamp(-127.0, 127.0);
        out[i] = q as i8;
    }
    (out, scale)
}

fn dequantize_rows_cpu(src_i8: &[i8], scales: &[f32], rows: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; shape_size(&[rows, d])];
    for r in 0..rows {
        let s = scales[r];
        let off = r * d;
        for i in 0..d {
            out[off + i] = src_i8[off + i] as f32 * s;
        }
    }
    out
}

#[cfg(feature = "cuda")]
fn quantize_rows_cuda(
    tensor_id: TensorId,
    rows: usize,
    d: usize,
    store: &TensorStore,
) -> Result<(Vec<i8>, Vec<f32>), String> {
    let dev = GpuDevice::instance();
    let in_ptr = store.dev_ptr(tensor_id);
    let q_buf: CudaSlice<i8> = dev
        .stream
        .alloc_zeros(rows * d)
        .map_err(|e| format!("alloc quant buffer failed: {e}"))?;
    let scale_buf: CudaSlice<f32> = dev
        .stream
        .alloc_zeros(rows)
        .map_err(|e| format!("alloc scale buffer failed: {e}"))?;
    let q_ptr = dev.ptr(&q_buf);
    let scale_ptr = dev.ptr(&scale_buf);
    let scale_func = dev.get_func("compute_rowwise_scale_f32");
    let quant_func = dev.get_func("quantize_rowwise_i8_f32");
    let row_grid = launch_cfg(rows as u32);
    let n = (rows * d) as u32;
    unsafe {
        dev.stream
            .launch_builder(scale_func)
            .arg(&in_ptr)
            .arg(&scale_ptr)
            .arg(&(rows as i32))
            .arg(&(d as i32))
            .launch(row_grid)
            .map_err(|e| format!("scale kernel launch failed: {e}"))?;
        dev.stream
            .launch_builder(quant_func)
            .arg(&in_ptr)
            .arg(&scale_ptr)
            .arg(&q_ptr)
            .arg(&(rows as i32))
            .arg(&(d as i32))
            .launch(launch_cfg(n))
            .map_err(|e| format!("quantize kernel launch failed: {e}"))?;
    }
    let q_host = dev
        .stream
        .memcpy_dtov(&q_buf)
        .map_err(|e| format!("copy quantized data failed: {e}"))?;
    let s_host = dev
        .stream
        .memcpy_dtov(&scale_buf)
        .map_err(|e| format!("copy quant scales failed: {e}"))?;
    Ok((q_host, s_host))
}

#[cfg(feature = "cuda")]
fn dequantize_rows_cuda(
    src_i8: &[i8],
    scales: &[f32],
    rows: usize,
    d: usize,
) -> Result<Vec<f32>, String> {
    let dev = GpuDevice::instance();
    let q_buf: CudaSlice<i8> = dev
        .stream
        .memcpy_stod(src_i8)
        .map_err(|e| format!("upload quantized rows failed: {e}"))?;
    let scale_buf: CudaSlice<f32> = dev
        .stream
        .memcpy_stod(scales)
        .map_err(|e| format!("upload scales failed: {e}"))?;
    let out_buf: CudaSlice<f32> = dev
        .stream
        .alloc_zeros(rows * d)
        .map_err(|e| format!("alloc dequant output failed: {e}"))?;

    let q_ptr = dev.ptr(&q_buf);
    let scale_ptr = dev.ptr(&scale_buf);
    let out_ptr = dev.ptr(&out_buf);
    let func = dev.get_func("dequantize_rowwise_i8_f32");
    let n = (rows * d) as u32;
    unsafe {
        dev.stream
            .launch_builder(func)
            .arg(&q_ptr)
            .arg(&scale_ptr)
            .arg(&out_ptr)
            .arg(&(rows as i32))
            .arg(&(d as i32))
            .launch(launch_cfg(n))
            .map_err(|e| format!("dequant kernel launch failed: {e}"))?;
    }
    dev.stream
        .memcpy_dtov(&out_buf)
        .map_err(|e| format!("download dequant output failed: {e}"))
}
