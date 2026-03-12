// Core autodiff and operators
export * from "./autodiff.js";
export * from "./operators.js";

// Scalar module
export * from "./scalar.js";
export * from "./scalar_functions.js";
export * from "./datasets.js"
export * from "./optimizer.js"
export * from "./module.js"

// Tensor module
export { Tensor, type TensorLike } from "./tensor.js";
export { TensorData, IndexingError, type Shape, type Strides, type Index, type OutIndex, type Storage, indexToPosition, toIndex, shapeProduct, strides, shapeBroadcast, broadcastIndex } from "./tensor_data.js";
export { TensorContext, TensorHistory, TensorFunction, Neg as TensorNeg, Sigmoid as TensorSigmoid, ReLU as TensorReLU, Log as TensorLog, Exp as TensorExp, Inv as TensorInv, Add as TensorAdd, Mul as TensorMul, LT as TensorLT, EQ as TensorEQ, Sum as TensorSum, Permute as TensorPermute, View as TensorView, Contiguous as TensorContiguous, Conv1d as TensorConv1d } from "./tensor_functions.js";
export * as tensorFunctions from "./tensor_functions.js";
export { tensorMap, tensorZip, tensorReduce, tensorConv1d } from "./tensor_ops.js";
export { fastTensorMap, fastTensorZip, fastTensorReduce, destroyPool } from "./fast_ops.js";

export * from "./datasets.js"
export * from "./module.js"