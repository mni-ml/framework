# mni-framework-cutile

A cuTile Rust backend for [@mni-ml/framework](https://github.com/mni-ml/framework).

This crate is a sibling of `src/native/` that re-implements a focused subset
of the framework's tensor ops using NVIDIA's cuTile Rust DSL
([NVlabs/cutile-rs](https://github.com/NVlabs/cutile-rs)) instead of the
default CUDA path.  It demonstrates that cuTile kernels can be dropped into
the framework's N-API surface alongside (or in place of) the existing cudarc
+ NVRTC backend.

## Status

Working.  Rust-level tests and N-API smoke tests pass on NVIDIA L40S (sm_89)
with CUDA 13.2 + cuTile `main`.

## What's implemented

Kernels (`src/kernels.rs`, in a `#[cutile::module]`):

| Op               | Kind                           |
|------------------|--------------------------------|
| `add/sub/mul/div`| Elementwise 1D, tile block 256 |
| `neg`            | Elementwise unary              |
| `mul_scalar`     | Elementwise × broadcast scalar |
| `saxpy`          | Fused `z = a·x + y` (2 GMEM reads, 1 GMEM write) |
| `relu`           | `max(0, x)` via `max_tile`     |
| `relu_backward`  | `grad where x > 0 else 0` via `gt_tile` + `select` |
| `sum_block`      | Per-block `reduce_sum(tile)` writing one scalar per pid; drives both passes of the CUB-style two-pass global reduction |
| `gemm`           | 2D tiled with `mma` accumulation, const generics `<BM, BN, BK, K>` |

Global reductions (`src/ops/reduce.rs`): `sum_all`, `mean_all` are a
strict CUB-style two-pass reduction driven by `sum_block`.  Pass 1 is a
multi-block launch that reduces `n` elements down to
`ceil(n / PASS1_BLOCK)` partials (one per block); pass 2 is a single-block
launch that reduces those partials to a single scalar.  When `n` is not a
multiple of `PASS1_BLOCK` — or when the pass-2 tile is wider than the
partials count — the out-of-range tile lanes are zero-padded automatically
by `Tensor::partition()` (`make_partition_view_padded(.., "zero", ..)` in
cuTile).  Zero is the additive identity, so the reduction result is
unaffected.  No host tail, no divisibility constraint on `n`.  Small
inputs (`n ≤ 4096`) skip pass 1 and go straight to a single zero-padded
pass-2 kernel.

TensorStore (`src/tensor.rs`): ID-keyed storage mirroring the main CUDA
backend — `zeros`, `ones`, `from_slice`, `rand`, `randn`, `to_host`, `free`,
with a free-ID recycler.

Runtime (`src/device.rs`): a single process-wide `CudaContext` + `CudaStream`
in a `OnceLock`, shared by every op.

## Building

```bash
# Rust-only (for `cargo test` / `cargo bench`):
cd src/cutile
cargo build --release

# With N-API bindings so the TypeScript loader can find it:
cargo build --release --features napi
cp target/release/libmni_framework_cutile.so \
   mni-framework-cutile.linux-x64-gnu.node
# ...or simply: npm run build:native:cutile  (from the framework root)
```

Requirements (matches cuTile's):

- Linux x86_64, NVIDIA GPU with sm_80+ (tested on sm_89)
- CUDA 13.2 toolkit at `CUDA_TOOLKIT_PATH`
- Rust 1.89+ (nightly optional)
- clang / libclang for the NVRTC bindgen step

The `.cargo/config.toml` pins `CUDA_TOOLKIT_PATH` and the `BINDGEN_EXTRA_CLANG_ARGS`.

## Using it from TypeScript

A new entry point `loadCutile()` in `src/native-loader.ts` returns the cuTile
N-API module directly — it is deliberately **not** slotted into the
default `loadNative()` priority list because it only implements a subset of
the framework's 50+ ops.  Intended use is opt-in, per-op:

```ts
import { loadCutile } from '@mni-ml/framework/native-loader';

const cutile = loadCutile();
const a = cutile.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
const b = cutile.fromFloat32(new Float32Array([1, 0, 0, 1, 1, 1]), [3, 2]);
const c = cutile.matmul(a, b);
console.log(cutile.toFloat32(c));  // Float32Array(4) [ 4, 5, 10, 11 ]
```

The napi exports are exactly the set implemented in `src/lib.rs`:

```
add, sub, mul, div, neg, mulScalar, saxpy, relu, reluBackward,
matmul, sumAll, meanAll, zeros, ones, randTensor, randnTensor,
fromFloat32, toFloat32, getScalar, tensorShape, tensorSize, freeTensor
```

## Tests

Rust-level correctness tests compare cuTile output against a CPU reference
(1e-4 abs/rel tolerance for elementwise, 2e-3 for GEMM).

```bash
cargo test --release --test correctness -- --test-threads=1
# 20 passed; 0 failed
```

N-API smoke tests drive the backend through the Node.js loader:

```bash
node test_napi.mjs    # loads .node directly
node test_loader.mjs  # goes through loadCutile()
```

## Benchmarks

Measured on an NVIDIA L40S (Ada Lovelace, sm_89) with cuTile on
CUDA 13.2.  CPU baseline is a single-threaded f32 loop, `--release`.
Numbers are median of 10 criterion samples.

### Elementwise `z = x + y`

| N           | cuTile       | CPU          | cuTile vs CPU |
|-------------|--------------|--------------|---------------|
| 1 024       | 22.4 µs      | 168 ns       | CPU wins (launch overhead) |
| 16 384      | 22.4 µs      | 1.98 µs      | CPU wins |
| 1 048 576   | 29.3 µs      | 217 µs       | **7.4× cuTile** |
| 4 194 304   | 49.0 µs      | 2.11 ms      | **43× cuTile** |

### Fused SAXPY `z = a·x + y`

| N           | cuTile       | CPU          | cuTile vs CPU |
|-------------|--------------|--------------|---------------|
| 1 048 576   | 30.0 µs      | 268 µs       | **8.9× cuTile** |
| 4 194 304   | 49.8 µs      | 2.23 ms      | **45× cuTile** |

At N = 4M, the kernel sustains ~85 Gelem/s — dominated by HBM bandwidth
rather than compute, as expected for a streaming op.

### GEMM `z = x @ y` (square, all f32)

| Shape          | cuTile   | CPU      | Speed-up | cuTile effective |
|----------------|----------|----------|----------|------------------|
| 64 × 64 × 64   | 43 µs    | 227 µs   | 5.3×     | 0.01 TFLOPs     |
| 128 × 128 × 128| 60 µs    | 2.90 ms  | 48×      | 0.07 TFLOPs     |
| 256 × 256 × 256| 114 µs   | 22.2 ms  | 195×     | 0.29 TFLOPs     |
| 512 × 512 × 512| 170 µs   | —        | —        | 1.58 TFLOPs     |

The naive `mma`-based GEMM in `kernels.rs` uses `BM = BN = 64`, `BK = 32`
tiles and leaves a lot of peak on the table (L40S fp32 peak is ~91 TFLOPs).
A production GEMM would need shared-memory staging, larger tiles, double
buffering, etc. — out of scope for this backend, which aims to prove the
wiring rather than push peak throughput.

### `sum_all` (global reduction, strict 2-pass)

| N           | cuTile   | CPU      | cuTile vs CPU |
|-------------|----------|----------|---------------|
| 16 384      | 53 µs    | 15 µs    | CPU wins (launch overhead) |
| 1 048 576   | 57 µs    | 976 µs   | **17× cuTile** |
| 4 194 304   | 64 µs    | 3.94 ms  | **61× cuTile** |

Pass 1 is a multi-block `sum_block::<2048>` launch that reduces `n` →
`ceil(n/2048)` partials; pass 2 is a single-block `sum_block` launch whose
tile size is the smallest power of 2 ≥ the partials count.  The in-block
reduction uses cuTile's built-in `reduce_sum` tile op; trailing lanes in
the last-block/pass-2 tile load 0.0 via the zero-padded partition.
Crossover vs the CPU sum is ~64K elements on L40S.

Reproduce:

```bash
cargo bench --bench kernels -- --quick
```

## Design notes

- **Sibling crate, not a feature flag.**  cuTile's `cuda-core` is not
  binary-compatible with the `cudarc` crate the default backend uses; sharing
  a process with a cudarc-owned CUDA context and a cuTile-owned one at the
  same time would require care.  Keeping this as its own `.node` sidesteps
  the issue: each user's program picks one.
- **Ops flatten inputs to 1D via `TensorView`.**  Elementwise kernels are
  compiled for a fixed rank of 1 (`Tensor<f32, { [-1] }>`).  Multi-dim
  tensors get a 1D `view(&[size])` at launch time, then the output is
  reshaped back into the logical shape.  No copies, no reallocations.
- **Tile block size picked to divide N.**  `pick_block` walks
  `[256, 128, … 1]` and returns the largest divisor of the element count.
  cuTile caches PTX per-generic, so each distinct block value costs one
  JIT compile at first launch, then is free.
- **GEMM shape constraints.**  `BM | M`, `BN | N`, `BK | K`, and `K` is a
  compile-time constant (each new K triggers a PTX recompile).  Typical ML
  shapes (64/128/256/...) all fit.  Non-divisible shapes are rejected at the
  op-wrapper level by the divisor walk rather than producing garbage via
  cuTile's zero-padded partitions.
- **Strict two-pass reductions.**  `sum_all` / `mean_all` are exactly two
  `sum_block` launches: multi-block (O(n) → O(n / PASS1_BLOCK)) then
  single-block (O(tiles) → 1).  The last block in pass 1 and the sole
  block in pass 2 are the usual "partial tile" case — `Tensor::partition()`
  returns a zero-padded view, so out-of-range lanes load the sum identity
  and the reduction stays correct for arbitrary `n`.  No host tail, no
  divisibility requirement, no recursion.

## Files

```
src/
├── device.rs    # shared CudaContext + stream via OnceLock
├── kernels.rs   # #[cutile::module] with all the tile kernels
├── tensor.rs    # ID-keyed TensorStore, creation / to_host / grad metadata
├── ops/
│   ├── elementwise.rs  # add / sub / mul / div / neg / mul_scalar / saxpy
│   ├── activation.rs   # relu / relu_backward
│   ├── matmul.rs       # GEMM with BM/BN/BK picker + generics
│   └── reduce.rs       # sum_all / mean_all (N-pass sum_block driver)
└── lib.rs       # pub modules + napi exports (feature-gated)

tests/correctness.rs      # 16 cross-checks vs CPU reference
benches/kernels.rs        # criterion benches vs CPU
test_napi.mjs             # N-API smoke test via direct require()
test_loader.mjs           # N-API smoke test via framework loadCutile()
```
