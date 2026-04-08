export * from "./core/index.js";
export * from "./nn/index.js";
export * from "./optimizer.js";
export * from "./datasets.js";
export { Scalar, ScalarHistory, type GradPair, type ScalarLike } from "./scalar.js";
export {
    IndexingError,
    TensorData,
    type Index,
    type OutIndex,
    type Shape as TensorDataShape,
    type Storage,
    type Strides,
} from "./tensor_data.js";

export * as nn from "./nn/index.js";
export * as extensions from "./extensions.js";
export * as toy from "./toy/index.js";

export function destroyPool(): void { /* no-op in native backend */ }
export function destroyDevice(): void { /* no-op in native backend */ }
