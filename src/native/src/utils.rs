/// Broadcast two shapes, returning the output shape.
/// Panics if shapes are incompatible.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let max_ndim = a.len().max(b.len());
    let mut out = vec![0usize; max_ndim];
    for i in 0..max_ndim {
        let da = if i < max_ndim - a.len() { 1 } else { a[i - (max_ndim - a.len())] };
        let db = if i < max_ndim - b.len() { 1 } else { b[i - (max_ndim - b.len())] };
        if da == db {
            out[i] = da;
        } else if da == 1 {
            out[i] = db;
        } else if db == 1 {
            out[i] = da;
        } else {
            panic!("shapes {:?} and {:?} are not broadcastable", a, b);
        }
    }
    out
}

/// Compute the flat index for a multi-dimensional coordinate given strides.
pub fn flat_index(coord: &[usize], strides: &[usize]) -> usize {
    coord.iter().zip(strides).map(|(&c, &s)| c * s).sum()
}

/// Convert a flat contiguous index into coordinates for the given shape.
pub fn to_coord(mut flat: usize, shape: &[usize], strides_out: &[usize]) -> Vec<usize> {
    let mut coord = vec![0usize; shape.len()];
    for i in 0..shape.len() {
        coord[i] = flat / strides_out[i];
        flat %= strides_out[i];
    }
    coord
}

/// Reduce a gradient from broadcast shape back to original shape by summing
/// over broadcast dimensions.
pub fn unbroadcast(grad_data: &[f32], grad_shape: &[usize], orig_shape: &[usize]) -> Vec<f32> {
    use crate::tensor::{compute_strides, shape_size};

    if grad_shape == orig_shape {
        return grad_data.to_vec();
    }

    let out_size = shape_size(orig_shape);
    let mut out = vec![0.0f32; out_size];
    let grad_strides = compute_strides(grad_shape);
    let orig_strides = compute_strides(orig_shape);
    let ndim = grad_shape.len();
    let orig_ndim = orig_shape.len();
    let offset = ndim - orig_ndim;

    let grad_size = shape_size(grad_shape);
    for i in 0..grad_size {
        let coord = to_coord(i, grad_shape, &grad_strides);
        let mut out_idx = 0;
        for d in 0..orig_ndim {
            let c = coord[d + offset];
            let c_clamped = if orig_shape[d] == 1 { 0 } else { c };
            out_idx += c_clamped * orig_strides[d];
        }
        out[out_idx] += grad_data[i];
    }
    out
}
