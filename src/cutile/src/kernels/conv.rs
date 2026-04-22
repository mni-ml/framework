//! 1D and 2D convolution cuTile kernels — port of `native/kernels/conv.cu`.
//!
//! Layout mirrors the CUDA reference exactly:
//!
//! - `inp`     is `[N, C_in, L]`        (1D) or `[N, C_in, H, W]`        (2D)
//! - `weight`  is `[C_out, C_in, K]`    (1D) or `[C_out, C_in, kH, kW]`  (2D)
//! - `out`     is `[N, C_out, L_out]`   (1D) or `[N, C_out, H_out, W_out]` (2D)
//!
//! One block per output element (fwd) or one block per input element
//! (bwd_input) or one block per weight element (bwd_weight), matching the
//! CUDA thread-per-element scheme.  All inner reductions over the
//! `C_in × K` (or `C_out × K`) window are expressed as a single `[CI, K]`
//! tile gather plus two `reduce_sum` passes.
//!
//! `stride` and `padding` are passed as runtime `i32` arguments — baking
//! them into const generics would explode the PTX cache across every
//! `(stride, padding)` combo a caller might use.
//!
//! Non-tile-aligned offsets along the spatial dimension (`l*stride - padding`)
//! don't fit the partition-load model, so these kernels use the
//! `pointer_to_tile` + `offset_tile` + `load_ptr_tko` gather pattern with a
//! boolean mask that matches the CUDA `if (il >= 0 && il < L)` bounds check
//! (and, for bwd_input, the `ol_raw >= 0 && ol_raw % stride == 0` condition).

#[cutile::module]
pub mod conv_kernels {
    use cutile::core::*;

    // ------------------------------------------------------------------
    // Conv1D
    // ------------------------------------------------------------------

    /// Per output element `(n, co, l_out)`:
    ///
    /// ```text
    ///   out[n, co, l_out] = Σ(ci, k) inp[n, ci, l_out*stride - pad + k]
    ///                                * weight[co, ci, k]
    /// ```
    ///
    /// Gathers a `[CI, K]` input window via `pointer_to_tile + offset_tile`
    /// with zero padding for OOB reads — cuTile partitions don't support
    /// element-granular offsets, and conv's sliding window isn't
    /// tile-aligned when `stride != K`.  The weight tile `[1, CI, K]` at
    /// `(co, 0, 0)` is a regular tile-aligned partition load since `CI`
    /// and `K` match the partition's tile extents.
    #[cutile::entry()]
    pub unsafe fn conv1d_forward<const CI: i32, const K: i32>(
        out: &mut Tensor<f32, { [1, 1, 1] }>,
        inp_ptr: *mut f32,
        weight: &Tensor<f32, { [-1, -1, -1] }>,
        l_in: i32,
        stride: i32,
        padding: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let n: i32 = pid.0;
        let co: i32 = pid.1;
        let l_out: i32 = pid.2;

        let w_part: Partition<f32, { [1, CI, K] }> = weight.partition(const_shape![1, CI, K]);
        let w_tile_3d: Tile<f32, { [1, CI, K] }> = w_part.load([co, 0i32, 0i32]);
        let w_tile: Tile<f32, { [CI, K] }> = w_tile_3d.reshape(const_shape![CI, K]);

        // Per-element input offsets: n*CI*L + ci*L + (l_out*stride - pad + k).
        let il_start: i32 = l_out * stride - padding;
        let base_off: i32 = n * CI * l_in;

        let ci_iota: Tile<i32, { [CI] }> = iota(const_shape![CI]);
        let l_t_ci: Tile<i32, { [CI] }> = l_in.broadcast(const_shape![CI]);
        let ci_off_1d: Tile<i32, { [CI] }> = ci_iota * l_t_ci;
        let ci_off: Tile<i32, { [CI, K] }> = ci_off_1d
            .reshape(const_shape![CI, 1])
            .broadcast(const_shape![CI, K]);

        let k_iota: Tile<i32, { [K] }> = iota(const_shape![K]);
        let il_start_t: Tile<i32, { [K] }> = il_start.broadcast(const_shape![K]);
        let il_1d: Tile<i32, { [K] }> = k_iota + il_start_t;
        let il_2d: Tile<i32, { [CI, K] }> = il_1d
            .reshape(const_shape![1, K])
            .broadcast(const_shape![CI, K]);

        let base_off_t: Tile<i32, { [CI, K] }> = base_off.broadcast(const_shape![CI, K]);
        let offs: Tile<i32, { [CI, K] }> = base_off_t + ci_off + il_2d;

        // Mask: per-column input l position must be in [0, L).
        let zero_ck: Tile<i32, { [CI, K] }> = constant(0i32, const_shape![CI, K]);
        let l_t_ck: Tile<i32, { [CI, K] }> = l_in.broadcast(const_shape![CI, K]);
        let mask: Tile<bool, { [CI, K] }> = ge_tile(il_2d, zero_ck) & lt_tile(il_2d, l_t_ck);

        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(inp_ptr);
        let base_2d: PointerTile<*mut f32, { [CI, K] }> = base
            .reshape(const_shape![1, 1])
            .broadcast(const_shape![CI, K]);
        let ptrs: PointerTile<*mut f32, { [CI, K] }> = base_2d.offset_tile(offs);

        let (inp_tile, _tok): (Tile<f32, { [CI, K] }>, Token) =
            load_ptr_tko(ptrs, "relaxed", "device", Some(mask), Some(0.0f32), None, None);

        let prod: Tile<f32, { [CI, K] }> = inp_tile * w_tile;
        let r1: Tile<f32, { [CI] }> = reduce_sum(prod, 1i32);
        let r2: Tile<f32, { [] }> = reduce_sum(r1, 0i32);
        out.store(r2.reshape(const_shape![1, 1, 1]));
    }

    /// Per input element `(n, ci, il)`:
    ///
    /// ```text
    ///   dinp[n, ci, il] = Σ(co, k) dout[n, co, (il + pad - k) / stride]
    ///                             * weight[co, ci, k]
    /// ```
    ///
    /// Contributions with `(il + pad - k) < 0`, `(il + pad - k) % stride != 0`,
    /// or `(il + pad - k) / stride >= L_out` are masked out — matching the
    /// CUDA `if (ol >= 0 && ol % stride == 0 && ol < L_out)` check.
    ///
    /// Both the `dout` and `weight` loads are pointer-tile gathers:
    /// `dout[n, :, ol(k)]` is a strided gather along the `co` axis with a
    /// per-column spatial index, and `weight[:, ci, :]` is a strided
    /// gather along the `co` axis (ci is fixed, not tile-aligned).
    #[cutile::entry()]
    pub unsafe fn conv1d_backward_input<const CO: i32, const K: i32>(
        dinp: &mut Tensor<f32, { [1, 1, 1] }>,
        dout_ptr: *mut f32,
        weight_ptr: *mut f32,
        c_in: i32,
        l_out: i32,
        stride: i32,
        padding: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let n: i32 = pid.0;
        let ci: i32 = pid.1;
        let il: i32 = pid.2;

        // weight[co, ci, k] offsets: co*(C_in*K) + ci*K + k, for (co, k) ∈ [CO, K].
        let co_iota: Tile<i32, { [CO] }> = iota(const_shape![CO]);
        let ci_k_stride: i32 = c_in * K;
        let ci_k_stride_t: Tile<i32, { [CO] }> = ci_k_stride.broadcast(const_shape![CO]);
        let co_off_1d: Tile<i32, { [CO] }> = co_iota * ci_k_stride_t;
        let co_off: Tile<i32, { [CO, K] }> = co_off_1d
            .reshape(const_shape![CO, 1])
            .broadcast(const_shape![CO, K]);

        let ci_off_scalar: i32 = ci * K;
        let ci_off_t: Tile<i32, { [CO, K] }> = ci_off_scalar.broadcast(const_shape![CO, K]);

        let k_iota: Tile<i32, { [K] }> = iota(const_shape![K]);
        let k_off: Tile<i32, { [CO, K] }> = k_iota
            .reshape(const_shape![1, K])
            .broadcast(const_shape![CO, K]);

        let w_offs: Tile<i32, { [CO, K] }> = co_off + ci_off_t + k_off;

        let w_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(weight_ptr);
        let w_base_2d: PointerTile<*mut f32, { [CO, K] }> = w_base
            .reshape(const_shape![1, 1])
            .broadcast(const_shape![CO, K]);
        let w_ptrs: PointerTile<*mut f32, { [CO, K] }> = w_base_2d.offset_tile(w_offs);
        let (w_tile, _wtok): (Tile<f32, { [CO, K] }>, Token) =
            load_ptr_tko(w_ptrs, "relaxed", "device", None, None, None, None);

        // dout[n, co, ol(k)] with ol_raw = il + pad - k; valid iff
        // ol_raw >= 0 && ol_raw % stride == 0 && (ol_raw/stride) < L_out.
        let il_pad: i32 = il + padding;
        let il_pad_t: Tile<i32, { [K] }> = il_pad.broadcast(const_shape![K]);
        let ol_raw: Tile<i32, { [K] }> = il_pad_t - k_iota;
        let stride_t: Tile<i32, { [K] }> = stride.broadcast(const_shape![K]);
        let zero_k: Tile<i32, { [K] }> = constant(0i32, const_shape![K]);
        let l_out_t_k: Tile<i32, { [K] }> = l_out.broadcast(const_shape![K]);

        let ol_div: Tile<i32, { [K] }> = ol_raw / stride_t;
        let ol_rem: Tile<i32, { [K] }> = ol_raw % stride_t;

        let ge0: Tile<bool, { [K] }> = ge_tile(ol_raw, zero_k);
        let modok: Tile<bool, { [K] }> = eq_tile(ol_rem, zero_k);
        let lt_lout: Tile<bool, { [K] }> = lt_tile(ol_div, l_out_t_k);
        let mask_k: Tile<bool, { [K] }> = ge0 & modok & lt_lout;
        let mask: Tile<bool, { [CO, K] }> = mask_k
            .reshape(const_shape![1, K])
            .broadcast(const_shape![CO, K]);

        // dout offsets per (co, k): n*(CO*L_out) + co*L_out + ol_div(k).
        let n_co_lout: i32 = n * CO * l_out;
        let n_co_lout_t: Tile<i32, { [CO, K] }> = n_co_lout.broadcast(const_shape![CO, K]);
        let lout_t_co: Tile<i32, { [CO] }> = l_out.broadcast(const_shape![CO]);
        let co_lout_1d: Tile<i32, { [CO] }> = co_iota * lout_t_co;
        let co_lout: Tile<i32, { [CO, K] }> = co_lout_1d
            .reshape(const_shape![CO, 1])
            .broadcast(const_shape![CO, K]);
        let ol_col: Tile<i32, { [CO, K] }> = ol_div
            .reshape(const_shape![1, K])
            .broadcast(const_shape![CO, K]);
        let d_offs: Tile<i32, { [CO, K] }> = n_co_lout_t + co_lout + ol_col;

        let d_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(dout_ptr);
        let d_base_2d: PointerTile<*mut f32, { [CO, K] }> = d_base
            .reshape(const_shape![1, 1])
            .broadcast(const_shape![CO, K]);
        let d_ptrs: PointerTile<*mut f32, { [CO, K] }> = d_base_2d.offset_tile(d_offs);
        let (d_tile, _dtok): (Tile<f32, { [CO, K] }>, Token) =
            load_ptr_tko(d_ptrs, "relaxed", "device", Some(mask), Some(0.0f32), None, None);

        let prod: Tile<f32, { [CO, K] }> = d_tile * w_tile;
        let r1: Tile<f32, { [CO] }> = reduce_sum(prod, 1i32);
        let r2: Tile<f32, { [] }> = reduce_sum(r1, 0i32);
        dinp.store(r2.reshape(const_shape![1, 1, 1]));
    }

    /// Per weight element `(co, ci, k)`:
    ///
    /// ```text
    ///   dweight[co, ci, k] = Σ(n, ol) dout[n, co, ol]
    ///                                * inp[n, ci, ol*stride - pad + k]
    ///                      over ol such that (ol*stride - pad + k) ∈ [0, L)
    /// ```
    ///
    /// Two-pass reduction: outer loop over `n` (runtime), inner tile of
    /// size `[BL]` along `l_out`.  Each inner iteration gathers a `[BL]`
    /// slice of `dout[n, co, :]` and a masked `[BL]` slice of
    /// `inp[n, ci, :]` at the stride-shifted positions, products the two,
    /// and accumulates into a scalar tile.
    #[cutile::entry()]
    pub unsafe fn conv1d_backward_weight<const BL: i32>(
        dweight: &mut Tensor<f32, { [1, 1, 1] }>,
        dout_ptr: *mut f32,
        inp_ptr: *mut f32,
        n_total: i32,
        c_in: i32,
        c_out: i32,
        l_in: i32,
        l_out: i32,
        stride: i32,
        padding: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let co: i32 = pid.0;
        let ci: i32 = pid.1;
        let k: i32 = pid.2;

        let num_l_tiles: i32 = ceil_div(l_out, BL);
        let mut acc: Tile<f32, { [] }> = constant(0.0f32, const_shape![]);

        let ol_iota: Tile<i32, { [BL] }> = iota(const_shape![BL]);
        let stride_t_bl: Tile<i32, { [BL] }> = stride.broadcast(const_shape![BL]);
        let zero_bl: Tile<i32, { [BL] }> = constant(0i32, const_shape![BL]);
        let l_in_t_bl: Tile<i32, { [BL] }> = l_in.broadcast(const_shape![BL]);
        let l_out_t_bl: Tile<i32, { [BL] }> = l_out.broadcast(const_shape![BL]);
        let k_minus_pad: i32 = k - padding;
        let k_pad_t: Tile<i32, { [BL] }> = k_minus_pad.broadcast(const_shape![BL]);

        for n in 0i32..n_total {
            for j in 0i32..num_l_tiles {
                let ol_base: i32 = j * BL;
                let ol_base_t: Tile<i32, { [BL] }> = ol_base.broadcast(const_shape![BL]);
                let ol_pos: Tile<i32, { [BL] }> = ol_base_t + ol_iota;

                // Per-lane validity: ol < L_out (tail of last tile),
                // and il = ol*stride + k - pad ∈ [0, L).
                let ol_lt: Tile<i32, { [BL] }> = ol_pos * stride_t_bl;
                let il_pos: Tile<i32, { [BL] }> = ol_lt + k_pad_t;
                let ol_valid: Tile<bool, { [BL] }> = lt_tile(ol_pos, l_out_t_bl);
                let il_valid: Tile<bool, { [BL] }> =
                    ge_tile(il_pos, zero_bl) & lt_tile(il_pos, l_in_t_bl);
                let mask_bl: Tile<bool, { [BL] }> = ol_valid & il_valid;

                // dout[n, co, ol] offset = n*CO*L_out + co*L_out + ol.
                let d_base_off: i32 = n * c_out * l_out + co * l_out;
                let d_base_t: Tile<i32, { [BL] }> = d_base_off.broadcast(const_shape![BL]);
                let d_offs: Tile<i32, { [BL] }> = d_base_t + ol_pos;

                let d_base_ptr: PointerTile<*mut f32, { [] }> = pointer_to_tile(dout_ptr);
                let d_base_bl: PointerTile<*mut f32, { [BL] }> = d_base_ptr
                    .reshape(const_shape![1])
                    .broadcast(const_shape![BL]);
                let d_ptrs: PointerTile<*mut f32, { [BL] }> = d_base_bl.offset_tile(d_offs);
                let (d_tile, _dt): (Tile<f32, { [BL] }>, Token) = load_ptr_tko(
                    d_ptrs,
                    "relaxed",
                    "device",
                    Some(mask_bl),
                    Some(0.0f32),
                    None,
                    None,
                );

                // inp[n, ci, il] offset = n*CI*L + ci*L + il.
                let i_base_off: i32 = n * c_in * l_in + ci * l_in;
                let i_base_t: Tile<i32, { [BL] }> = i_base_off.broadcast(const_shape![BL]);
                let i_offs: Tile<i32, { [BL] }> = i_base_t + il_pos;

                let i_base_ptr: PointerTile<*mut f32, { [] }> = pointer_to_tile(inp_ptr);
                let i_base_bl: PointerTile<*mut f32, { [BL] }> = i_base_ptr
                    .reshape(const_shape![1])
                    .broadcast(const_shape![BL]);
                let i_ptrs: PointerTile<*mut f32, { [BL] }> = i_base_bl.offset_tile(i_offs);
                let (i_tile, _it): (Tile<f32, { [BL] }>, Token) = load_ptr_tko(
                    i_ptrs,
                    "relaxed",
                    "device",
                    Some(mask_bl),
                    Some(0.0f32),
                    None,
                    None,
                );

                let prod: Tile<f32, { [BL] }> = d_tile * i_tile;
                let chunk: Tile<f32, { [] }> = reduce_sum(prod, 0i32);
                acc = acc + chunk;
            }
        }

        dweight.store(acc.reshape(const_shape![1, 1, 1]));
    }

    // ------------------------------------------------------------------
    // Conv2D
    // ------------------------------------------------------------------

    /// Per output element `(n, co, oh, ow)` — grid is flattened as
    /// `(N*C_out, H_out, W_out)` to fit in the 3D block id: `pid.0`
    /// encodes `n*C_out + co`, `pid.1 = oh`, `pid.2 = ow`.
    ///
    /// ```text
    ///   out[n, co, oh, ow] = Σ(ci, kh, kw) inp[n, ci, oh*s - p + kh, ow*s - p + kw]
    ///                                     * weight[co, ci, kh, kw]
    /// ```
    ///
    /// Gathers a `[CI, KH*KW]` pointer tile over the input window.  We
    /// flatten the `(kh, kw)` axes into a single `KHW = KH*KW` axis for
    /// the gather, then reshape the weight tile to match.
    #[cutile::entry()]
    pub unsafe fn conv2d_forward<const CI: i32, const KH: i32, const KW: i32>(
        out: &mut Tensor<f32, { [1, 1, 1, 1] }>,
        inp_ptr: *mut f32,
        weight: &Tensor<f32, { [-1, -1, -1, -1] }>,
        c_out: i32,
        h_in: i32,
        w_in: i32,
        stride: i32,
        padding: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let n_co: i32 = pid.0;
        let n: i32 = n_co / c_out;
        let co: i32 = n_co % c_out;
        let oh: i32 = pid.1;
        let ow: i32 = pid.2;

        let w_part: Partition<f32, { [1, CI, KH, KW] }> =
            weight.partition(const_shape![1, CI, KH, KW]);
        let w_tile_4d: Tile<f32, { [1, CI, KH, KW] }> = w_part.load([co, 0i32, 0i32, 0i32]);
        let w_tile: Tile<f32, { [CI, KH, KW] }> = w_tile_4d.reshape(const_shape![CI, KH, KW]);

        let ih_start: i32 = oh * stride - padding;
        let iw_start: i32 = ow * stride - padding;
        let base_off: i32 = n * CI * h_in * w_in;
        let hw: i32 = h_in * w_in;

        // Per-(ci, kh, kw) offsets: base + ci*H*W + (ih_start+kh)*W + (iw_start+kw).
        let ci_iota: Tile<i32, { [CI] }> = iota(const_shape![CI]);
        let hw_t_ci: Tile<i32, { [CI] }> = hw.broadcast(const_shape![CI]);
        let ci_off_1d: Tile<i32, { [CI] }> = ci_iota * hw_t_ci;
        let ci_off: Tile<i32, { [CI, KH, KW] }> = ci_off_1d
            .reshape(const_shape![CI, 1, 1])
            .broadcast(const_shape![CI, KH, KW]);

        let kh_iota: Tile<i32, { [KH] }> = iota(const_shape![KH]);
        let ih_start_t: Tile<i32, { [KH] }> = ih_start.broadcast(const_shape![KH]);
        let ih_1d: Tile<i32, { [KH] }> = kh_iota + ih_start_t;
        let ih_3d: Tile<i32, { [CI, KH, KW] }> = ih_1d
            .reshape(const_shape![1, KH, 1])
            .broadcast(const_shape![CI, KH, KW]);
        let w_in_t_chw: Tile<i32, { [CI, KH, KW] }> = w_in.broadcast(const_shape![CI, KH, KW]);
        let ih_row: Tile<i32, { [CI, KH, KW] }> = ih_3d * w_in_t_chw;

        let kw_iota: Tile<i32, { [KW] }> = iota(const_shape![KW]);
        let iw_start_t: Tile<i32, { [KW] }> = iw_start.broadcast(const_shape![KW]);
        let iw_1d: Tile<i32, { [KW] }> = kw_iota + iw_start_t;
        let iw_3d: Tile<i32, { [CI, KH, KW] }> = iw_1d
            .reshape(const_shape![1, 1, KW])
            .broadcast(const_shape![CI, KH, KW]);

        let base_off_t: Tile<i32, { [CI, KH, KW] }> =
            base_off.broadcast(const_shape![CI, KH, KW]);
        let offs: Tile<i32, { [CI, KH, KW] }> = base_off_t + ci_off + ih_row + iw_3d;

        // Mask: (ih, iw) ∈ [0, H) × [0, W).
        let zero_chw: Tile<i32, { [CI, KH, KW] }> = constant(0i32, const_shape![CI, KH, KW]);
        let h_t_chw: Tile<i32, { [CI, KH, KW] }> = h_in.broadcast(const_shape![CI, KH, KW]);
        let h_ok: Tile<bool, { [CI, KH, KW] }> = ge_tile(ih_3d, zero_chw) & lt_tile(ih_3d, h_t_chw);
        let w_ok: Tile<bool, { [CI, KH, KW] }> = ge_tile(iw_3d, zero_chw) & lt_tile(iw_3d, w_in_t_chw);
        let mask: Tile<bool, { [CI, KH, KW] }> = h_ok & w_ok;

        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(inp_ptr);
        let base_3d: PointerTile<*mut f32, { [CI, KH, KW] }> = base
            .reshape(const_shape![1, 1, 1])
            .broadcast(const_shape![CI, KH, KW]);
        let ptrs: PointerTile<*mut f32, { [CI, KH, KW] }> = base_3d.offset_tile(offs);
        let (inp_tile, _tok): (Tile<f32, { [CI, KH, KW] }>, Token) =
            load_ptr_tko(ptrs, "relaxed", "device", Some(mask), Some(0.0f32), None, None);

        let prod: Tile<f32, { [CI, KH, KW] }> = inp_tile * w_tile;
        let r1: Tile<f32, { [CI, KH] }> = reduce_sum(prod, 2i32);
        let r2: Tile<f32, { [CI] }> = reduce_sum(r1, 1i32);
        let r3: Tile<f32, { [] }> = reduce_sum(r2, 0i32);
        out.store(r3.reshape(const_shape![1, 1, 1, 1]));
    }

    /// Per input element `(n, ci, ih, iw)` — grid is
    /// `(N*C_in, H, W)` with `pid.0 = n*C_in + ci`, `pid.1 = ih`,
    /// `pid.2 = iw`.
    ///
    /// ```text
    ///   oh_raw = ih + pad - kh ; ow_raw = iw + pad - kw
    ///   dinp[n, ci, ih, iw] = Σ(co, kh, kw) dout[n, co, oh_raw/s, ow_raw/s]
    ///                                      * weight[co, ci, kh, kw]
    ///      where oh_raw, ow_raw are ≥ 0 and ≡ 0 (mod s) and oh, ow < out bounds
    /// ```
    #[cutile::entry()]
    pub unsafe fn conv2d_backward_input<const CO: i32, const KH: i32, const KW: i32>(
        dinp: &mut Tensor<f32, { [1, 1, 1, 1] }>,
        dout_ptr: *mut f32,
        weight_ptr: *mut f32,
        c_in: i32,
        h_out: i32,
        w_out: i32,
        stride: i32,
        padding: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let n_ci: i32 = pid.0;
        let n: i32 = n_ci / c_in;
        let ci: i32 = n_ci % c_in;
        let ih: i32 = pid.1;
        let iw: i32 = pid.2;

        // weight[co, ci, kh, kw] offsets: co*(CI*KH*KW) + ci*(KH*KW) + kh*KW + kw.
        let khkw: i32 = KH * KW;
        let co_iota: Tile<i32, { [CO] }> = iota(const_shape![CO]);
        let ci_khw_stride: i32 = c_in * khkw;
        let ci_khw_t: Tile<i32, { [CO] }> = ci_khw_stride.broadcast(const_shape![CO]);
        let co_off_1d: Tile<i32, { [CO] }> = co_iota * ci_khw_t;
        let co_off: Tile<i32, { [CO, KH, KW] }> = co_off_1d
            .reshape(const_shape![CO, 1, 1])
            .broadcast(const_shape![CO, KH, KW]);

        let ci_off_scalar: i32 = ci * khkw;
        let ci_off_t: Tile<i32, { [CO, KH, KW] }> =
            ci_off_scalar.broadcast(const_shape![CO, KH, KW]);

        let kh_iota: Tile<i32, { [KH] }> = iota(const_shape![KH]);
        let kw_const: i32 = KW;
        let kw_t_kh: Tile<i32, { [KH] }> = kw_const.broadcast(const_shape![KH]);
        let kh_w_1d: Tile<i32, { [KH] }> = kh_iota * kw_t_kh;
        let kh_w_off: Tile<i32, { [CO, KH, KW] }> = kh_w_1d
            .reshape(const_shape![1, KH, 1])
            .broadcast(const_shape![CO, KH, KW]);

        let kw_iota: Tile<i32, { [KW] }> = iota(const_shape![KW]);
        let kw_off: Tile<i32, { [CO, KH, KW] }> = kw_iota
            .reshape(const_shape![1, 1, KW])
            .broadcast(const_shape![CO, KH, KW]);

        let w_offs: Tile<i32, { [CO, KH, KW] }> = co_off + ci_off_t + kh_w_off + kw_off;

        let w_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(weight_ptr);
        let w_base_3d: PointerTile<*mut f32, { [CO, KH, KW] }> = w_base
            .reshape(const_shape![1, 1, 1])
            .broadcast(const_shape![CO, KH, KW]);
        let w_ptrs: PointerTile<*mut f32, { [CO, KH, KW] }> = w_base_3d.offset_tile(w_offs);
        let (w_tile, _wtok): (Tile<f32, { [CO, KH, KW] }>, Token) =
            load_ptr_tko(w_ptrs, "relaxed", "device", None, None, None, None);

        // For each (kh, kw): oh_raw = ih + pad - kh, ow_raw = iw + pad - kw.
        // Valid iff oh_raw, ow_raw ≥ 0 and ≡ 0 (mod stride) and /stride < H_out/W_out.
        let ih_pad: i32 = ih + padding;
        let iw_pad: i32 = iw + padding;

        let ih_pad_t_kh: Tile<i32, { [KH] }> = ih_pad.broadcast(const_shape![KH]);
        let oh_raw_1d: Tile<i32, { [KH] }> = ih_pad_t_kh - kh_iota;
        let stride_t_kh: Tile<i32, { [KH] }> = stride.broadcast(const_shape![KH]);
        let zero_kh: Tile<i32, { [KH] }> = constant(0i32, const_shape![KH]);
        let h_out_t_kh: Tile<i32, { [KH] }> = h_out.broadcast(const_shape![KH]);
        let oh_div_1d: Tile<i32, { [KH] }> = oh_raw_1d / stride_t_kh;
        let oh_rem_1d: Tile<i32, { [KH] }> = oh_raw_1d % stride_t_kh;
        let oh_ok_1d: Tile<bool, { [KH] }> = ge_tile(oh_raw_1d, zero_kh)
            & eq_tile(oh_rem_1d, zero_kh)
            & lt_tile(oh_div_1d, h_out_t_kh);

        let iw_pad_t_kw: Tile<i32, { [KW] }> = iw_pad.broadcast(const_shape![KW]);
        let ow_raw_1d: Tile<i32, { [KW] }> = iw_pad_t_kw - kw_iota;
        let stride_t_kw: Tile<i32, { [KW] }> = stride.broadcast(const_shape![KW]);
        let zero_kw: Tile<i32, { [KW] }> = constant(0i32, const_shape![KW]);
        let w_out_t_kw: Tile<i32, { [KW] }> = w_out.broadcast(const_shape![KW]);
        let ow_div_1d: Tile<i32, { [KW] }> = ow_raw_1d / stride_t_kw;
        let ow_rem_1d: Tile<i32, { [KW] }> = ow_raw_1d % stride_t_kw;
        let ow_ok_1d: Tile<bool, { [KW] }> = ge_tile(ow_raw_1d, zero_kw)
            & eq_tile(ow_rem_1d, zero_kw)
            & lt_tile(ow_div_1d, w_out_t_kw);

        let oh_ok_3d: Tile<bool, { [CO, KH, KW] }> = oh_ok_1d
            .reshape(const_shape![1, KH, 1])
            .broadcast(const_shape![CO, KH, KW]);
        let ow_ok_3d: Tile<bool, { [CO, KH, KW] }> = ow_ok_1d
            .reshape(const_shape![1, 1, KW])
            .broadcast(const_shape![CO, KH, KW]);
        let mask: Tile<bool, { [CO, KH, KW] }> = oh_ok_3d & ow_ok_3d;

        // dout[n, co, oh, ow] offset = n*(CO*H_out*W_out) + co*(H_out*W_out)
        //                              + oh_div*W_out + ow_div.
        let hwout: i32 = h_out * w_out;
        let n_cohw: i32 = n * CO * hwout;
        let n_cohw_t: Tile<i32, { [CO, KH, KW] }> = n_cohw.broadcast(const_shape![CO, KH, KW]);
        let hwout_t_co: Tile<i32, { [CO] }> = hwout.broadcast(const_shape![CO]);
        let co_hw_1d: Tile<i32, { [CO] }> = co_iota * hwout_t_co;
        let co_hw: Tile<i32, { [CO, KH, KW] }> = co_hw_1d
            .reshape(const_shape![CO, 1, 1])
            .broadcast(const_shape![CO, KH, KW]);

        let wout_t_kh: Tile<i32, { [KH] }> = w_out.broadcast(const_shape![KH]);
        let oh_row_1d: Tile<i32, { [KH] }> = oh_div_1d * wout_t_kh;
        let oh_row_3d: Tile<i32, { [CO, KH, KW] }> = oh_row_1d
            .reshape(const_shape![1, KH, 1])
            .broadcast(const_shape![CO, KH, KW]);

        let ow_col_3d: Tile<i32, { [CO, KH, KW] }> = ow_div_1d
            .reshape(const_shape![1, 1, KW])
            .broadcast(const_shape![CO, KH, KW]);

        let d_offs: Tile<i32, { [CO, KH, KW] }> = n_cohw_t + co_hw + oh_row_3d + ow_col_3d;

        let d_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(dout_ptr);
        let d_base_3d: PointerTile<*mut f32, { [CO, KH, KW] }> = d_base
            .reshape(const_shape![1, 1, 1])
            .broadcast(const_shape![CO, KH, KW]);
        let d_ptrs: PointerTile<*mut f32, { [CO, KH, KW] }> = d_base_3d.offset_tile(d_offs);
        let (d_tile, _dtok): (Tile<f32, { [CO, KH, KW] }>, Token) =
            load_ptr_tko(d_ptrs, "relaxed", "device", Some(mask), Some(0.0f32), None, None);

        let prod: Tile<f32, { [CO, KH, KW] }> = d_tile * w_tile;
        let r1: Tile<f32, { [CO, KH] }> = reduce_sum(prod, 2i32);
        let r2: Tile<f32, { [CO] }> = reduce_sum(r1, 1i32);
        let r3: Tile<f32, { [] }> = reduce_sum(r2, 0i32);
        dinp.store(r3.reshape(const_shape![1, 1, 1, 1]));
    }

    /// Per weight element `(co, ci, kh, kw)` — grid is
    /// `(C_out*C_in, KH, KW)` with `pid.0 = co*C_in + ci`.
    ///
    /// ```text
    ///   dweight[co, ci, kh, kw] = Σ(n, oh, ow) dout[n, co, oh, ow]
    ///                                         * inp[n, ci, oh*s - p + kh, ow*s - p + kw]
    /// ```
    ///
    /// Double loop over `n` and `oh`, with an inner `[BW]` tile along
    /// `ow`.  Mask handles tail of last tile + `ih/iw` input bounds.
    #[cutile::entry()]
    pub unsafe fn conv2d_backward_weight<const BW: i32>(
        dweight: &mut Tensor<f32, { [1, 1, 1, 1] }>,
        dout_ptr: *mut f32,
        inp_ptr: *mut f32,
        n_total: i32,
        c_in: i32,
        c_out: i32,
        h_in: i32,
        w_in: i32,
        h_out: i32,
        w_out: i32,
        stride: i32,
        padding: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let co_ci: i32 = pid.0;
        let co: i32 = co_ci / c_in;
        let ci: i32 = co_ci % c_in;
        let kh: i32 = pid.1;
        let kw: i32 = pid.2;

        let num_w_tiles: i32 = ceil_div(w_out, BW);
        let mut acc: Tile<f32, { [] }> = constant(0.0f32, const_shape![]);

        let ow_iota: Tile<i32, { [BW] }> = iota(const_shape![BW]);
        let stride_t_bw: Tile<i32, { [BW] }> = stride.broadcast(const_shape![BW]);
        let zero_bw: Tile<i32, { [BW] }> = constant(0i32, const_shape![BW]);
        let w_in_t_bw: Tile<i32, { [BW] }> = w_in.broadcast(const_shape![BW]);
        let w_out_t_bw: Tile<i32, { [BW] }> = w_out.broadcast(const_shape![BW]);
        let kw_pad: i32 = kw - padding;
        let kw_pad_t_bw: Tile<i32, { [BW] }> = kw_pad.broadcast(const_shape![BW]);

        let hwout: i32 = h_out * w_out;
        let hwin: i32 = h_in * w_in;

        for n in 0i32..n_total {
            for oh in 0i32..h_out {
                let ih_raw: i32 = oh * stride + kh - padding;
                if ih_raw < 0 || ih_raw >= h_in {
                    continue;
                }
                for j in 0i32..num_w_tiles {
                    let ow_base: i32 = j * BW;
                    let ow_base_t: Tile<i32, { [BW] }> = ow_base.broadcast(const_shape![BW]);
                    let ow_pos: Tile<i32, { [BW] }> = ow_base_t + ow_iota;

                    let ow_lt: Tile<i32, { [BW] }> = ow_pos * stride_t_bw;
                    let iw_pos: Tile<i32, { [BW] }> = ow_lt + kw_pad_t_bw;
                    let ow_valid: Tile<bool, { [BW] }> = lt_tile(ow_pos, w_out_t_bw);
                    let iw_valid: Tile<bool, { [BW] }> =
                        ge_tile(iw_pos, zero_bw) & lt_tile(iw_pos, w_in_t_bw);
                    let mask_bw: Tile<bool, { [BW] }> = ow_valid & iw_valid;

                    // dout[n, co, oh, ow] linear offset.
                    let d_row_base: i32 = n * c_out * hwout + co * hwout + oh * w_out;
                    let d_row_t: Tile<i32, { [BW] }> = d_row_base.broadcast(const_shape![BW]);
                    let d_offs: Tile<i32, { [BW] }> = d_row_t + ow_pos;

                    let d_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(dout_ptr);
                    let d_base_bw: PointerTile<*mut f32, { [BW] }> = d_base
                        .reshape(const_shape![1])
                        .broadcast(const_shape![BW]);
                    let d_ptrs: PointerTile<*mut f32, { [BW] }> = d_base_bw.offset_tile(d_offs);
                    let (d_tile, _dt): (Tile<f32, { [BW] }>, Token) = load_ptr_tko(
                        d_ptrs,
                        "relaxed",
                        "device",
                        Some(mask_bw),
                        Some(0.0f32),
                        None,
                        None,
                    );

                    // inp[n, ci, ih_raw, iw] linear offset.
                    let i_row_base: i32 = n * c_in * hwin + ci * hwin + ih_raw * w_in;
                    let i_row_t: Tile<i32, { [BW] }> = i_row_base.broadcast(const_shape![BW]);
                    let i_offs: Tile<i32, { [BW] }> = i_row_t + iw_pos;

                    let i_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(inp_ptr);
                    let i_base_bw: PointerTile<*mut f32, { [BW] }> = i_base
                        .reshape(const_shape![1])
                        .broadcast(const_shape![BW]);
                    let i_ptrs: PointerTile<*mut f32, { [BW] }> = i_base_bw.offset_tile(i_offs);
                    let (i_tile, _it): (Tile<f32, { [BW] }>, Token) = load_ptr_tko(
                        i_ptrs,
                        "relaxed",
                        "device",
                        Some(mask_bw),
                        Some(0.0f32),
                        None,
                        None,
                    );

                    let prod: Tile<f32, { [BW] }> = d_tile * i_tile;
                    let chunk: Tile<f32, { [] }> = reduce_sum(prod, 0i32);
                    acc = acc + chunk;
                }
            }
        }

        dweight.store(acc.reshape(const_shape![1, 1, 1, 1]));
    }
}

pub use conv_kernels::{
    conv1d_backward_input, conv1d_backward_weight, conv1d_forward, conv2d_backward_input,
    conv2d_backward_weight, conv2d_forward,
};
