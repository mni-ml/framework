use smallvec::SmallVec;
use crate::ops::data::IntStore;
use crate::tensor::{TensorId, TensorStore};

/// What was saved during forward for use in backward.
#[derive(Clone)]
pub enum SavedContext {
    None,
    Tensor(TensorId),
    Tensors(SmallVec<[TensorId; 4]>),
    TensorAndScalar(TensorId, f32),
    TensorsAndShape(SmallVec<[TensorId; 4]>, Vec<usize>),
    TensorAndShape(TensorId, Vec<usize>),
    Indices(Vec<usize>, usize, usize, TensorId), // indices, batch, seq_len, weight_id
    GpuIndices(usize, usize, usize, TensorId),  // int_buf_id, dim1, dim2, aux_tensor_id
    FlashAttention {
        q: TensorId, k: TensorId, v: TensorId,
        out: TensorId, lse: TensorId,
        scale: f32, s: usize, d: usize, causal: bool,
    },
    DropoutMask(TensorId, f32),
    Shape(Vec<usize>),
    Permutation(Vec<usize>, Vec<usize>), // order, original_shape
    ScalarAndTensor(f32, TensorId),
}

#[derive(Clone, Copy)]
pub enum BackwardOp {
    Add,
    Mul,
    Sub,
    Neg,
    MulScalar,
    Exp,
    Log,
    MatMul,
    Gelu,
    Relu,
    Sum,
    Mean,
    Max,
    View,
    Permute,
    Contiguous,
    Softmax,
    LayerNorm,
    Embedding,
    CrossEntropy,
    EmbeddingGpu,
    CrossEntropyGpu,
    FlashAttention,
    ResidualLayerNorm,
    BiasGelu,
    Dropout,
    Div,
    Sigmoid,
    Pow,
    Conv1d,
    Conv2d,
    AvgPool2d,
    MaxPool2d,
}

#[derive(Clone)]
pub struct TapeEntry {
    pub op: BackwardOp,
    pub output_id: TensorId,
    pub input_ids: SmallVec<[TensorId; 2]>,
    pub saved: SavedContext,
}

pub struct Tape {
    entries: Vec<TapeEntry>,
    enabled: bool,
}

impl Tape {
    pub fn new() -> Self {
        Self { entries: Vec::new(), enabled: true }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn record(&mut self, entry: TapeEntry) {
        if self.enabled {
            self.entries.push(entry);
        }
    }

    pub fn backward(self, loss_id: TensorId, store: &mut TensorStore, int_store: &IntStore) {
        use std::collections::{HashMap, HashSet};

        // Build adjacency: output -> entry index
        let mut output_to_entry: HashMap<TensorId, usize> = HashMap::new();
        for (i, e) in self.entries.iter().enumerate() {
            output_to_entry.insert(e.output_id, i);
        }

        // Find all tensors that contribute to loss via DFS
        let mut relevant: HashSet<TensorId> = HashSet::new();
        let mut stack = vec![loss_id];
        while let Some(tid) = stack.pop() {
            if !relevant.insert(tid) { continue; }
            if let Some(&idx) = output_to_entry.get(&tid) {
                for &inp in &self.entries[idx].input_ids {
                    stack.push(inp);
                }
            }
        }

        // Topological sort (reverse DFS post-order)
        let mut visited: HashSet<TensorId> = HashSet::new();
        let mut order: Vec<TensorId> = Vec::new();
        fn dfs(
            tid: TensorId,
            output_to_entry: &HashMap<TensorId, usize>,
            entries: &[TapeEntry],
            relevant: &HashSet<TensorId>,
            visited: &mut HashSet<TensorId>,
            order: &mut Vec<TensorId>,
        ) {
            if !visited.insert(tid) { return; }
            if let Some(&idx) = output_to_entry.get(&tid) {
                for &inp in &entries[idx].input_ids {
                    if relevant.contains(&inp) {
                        dfs(inp, output_to_entry, entries, relevant, visited, order);
                    }
                }
            }
            order.push(tid);
        }
        dfs(loss_id, &output_to_entry, &self.entries, &relevant, &mut visited, &mut order);
        order.reverse();

        // Initialize grad for loss = ones_like
        let mut grads: HashMap<TensorId, TensorId> = HashMap::new();
        let loss_grad = store.ones_like(loss_id);
        grads.insert(loss_id, loss_grad);

        // Backward pass
        for tid in &order {
            let grad_id = match grads.get(tid) {
                Some(&g) => g,
                None => continue,
            };

            if let Some(&entry_idx) = output_to_entry.get(tid) {
                let entry = self.entries[entry_idx].clone();
                let input_grads = dispatch_backward(&entry, grad_id, store, int_store);

                for (inp_id, inp_grad) in entry.input_ids.iter().zip(input_grads) {
                    if let Some(ig) = inp_grad {
                        if store.get(*inp_id).requires_grad {
                            store.accumulate_grad(*inp_id, ig);
                        }
                        if let Some(&existing) = grads.get(inp_id) {
                            store.add_inplace(existing, ig);
                        } else {
                            grads.insert(*inp_id, ig);
                        }
                    }
                }
            }
        }
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

fn dispatch_backward(
    entry: &TapeEntry,
    grad_id: TensorId,
    store: &mut TensorStore,
    int_store: &IntStore,
) -> Vec<Option<TensorId>> {
    use crate::ops;
    match entry.op {
        BackwardOp::Add => ops::elementwise::add_backward(grad_id, &entry.saved, store),
        BackwardOp::Mul => ops::elementwise::mul_backward(grad_id, &entry.saved, store),
        BackwardOp::Sub => ops::elementwise::sub_backward(grad_id, &entry.saved, store),
        BackwardOp::Neg => ops::elementwise::neg_backward(grad_id, &entry.saved, store),
        BackwardOp::MulScalar => ops::elementwise::mul_scalar_backward(grad_id, &entry.saved, store),
        BackwardOp::Exp => ops::elementwise::exp_backward(grad_id, &entry.saved, store),
        BackwardOp::Log => ops::elementwise::log_backward(grad_id, &entry.saved, store),
        BackwardOp::MatMul => ops::matmul::matmul_backward(grad_id, &entry.saved, store),
        BackwardOp::Gelu => ops::activation::gelu_backward(grad_id, &entry.saved, store),
        BackwardOp::Relu => ops::activation::relu_backward(grad_id, &entry.saved, store),
        BackwardOp::Sum => ops::reduce::sum_backward(grad_id, &entry.saved, store),
        BackwardOp::Mean => ops::reduce::mean_backward(grad_id, &entry.saved, store),
        BackwardOp::Max => ops::reduce::max_backward(grad_id, &entry.saved, store),
        BackwardOp::View => ops::layout::view_backward(grad_id, &entry.saved, store),
        BackwardOp::Permute => ops::layout::permute_backward(grad_id, &entry.saved, store),
        BackwardOp::Contiguous => ops::layout::contiguous_backward(grad_id, &entry.saved, store),
        BackwardOp::Softmax => ops::norm::softmax_backward(grad_id, &entry.saved, store),
        BackwardOp::LayerNorm => ops::norm::layernorm_backward(grad_id, &entry.saved, store),
        BackwardOp::Embedding => ops::embedding::embedding_backward(grad_id, &entry.saved, store),
        BackwardOp::CrossEntropy => ops::loss::cross_entropy_backward(grad_id, &entry.saved, store),
        BackwardOp::EmbeddingGpu => ops::embedding::embedding_backward_gpu(grad_id, &entry.saved, int_store, store),
        BackwardOp::CrossEntropyGpu => ops::loss::cross_entropy_backward_gpu(grad_id, &entry.saved, int_store, store),
        BackwardOp::FlashAttention => ops::attention::flash_attention_backward(grad_id, &entry.saved, store),
        BackwardOp::ResidualLayerNorm => ops::fused::residual_layernorm_backward(grad_id, &entry.saved, store),
        BackwardOp::BiasGelu => ops::fused::bias_gelu_backward(grad_id, &entry.saved, store),
        BackwardOp::Dropout => ops::dropout::dropout_backward(grad_id, &entry.saved, store),
        BackwardOp::Div => ops::elementwise::div_backward(grad_id, &entry.saved, store),
        BackwardOp::Sigmoid => ops::activation::sigmoid_backward(grad_id, &entry.saved, store),
        BackwardOp::Pow => ops::elementwise::pow_backward(grad_id, &entry.saved, store),
        BackwardOp::Conv1d => ops::conv::conv1d_backward(grad_id, &entry.saved, store),
        BackwardOp::Conv2d => ops::conv::conv2d_backward(grad_id, &entry.saved, store),
        BackwardOp::AvgPool2d => ops::pooling::avgpool2d_backward(grad_id, &entry.saved, store),
        BackwardOp::MaxPool2d => ops::pooling::maxpool2d_backward(grad_id, &entry.saved, store),
    }
}
