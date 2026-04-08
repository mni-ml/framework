import { Tensor, native } from "../../core/tensor.js";

function flattenClassTargets(targets: number[][] | Tensor, vocabSize: number): number[] {
    if (targets instanceof Tensor) {
        const data = targets.toFloat32();
        const classes: number[] = [];

        for (let offset = 0; offset < data.length; offset += vocabSize) {
            let maxIndex = 0;
            let maxValue = -Infinity;

            for (let i = 0; i < vocabSize; i++) {
                const value = data[offset + i]!;
                if (value > maxValue) {
                    maxValue = value;
                    maxIndex = i;
                }
            }

            classes.push(maxIndex);
        }

        return classes;
    }

    return targets.flat();
}

export function softmax(x: Tensor, dim: number = -1): Tensor {
    return new Tensor(native.softmaxOp(x._id, dim));
}

export function gelu(x: Tensor): Tensor {
    return new Tensor(native.gelu(x._id));
}

export function dropout(x: Tensor, rate: number = 0.0, inference: boolean = false): Tensor {
    if (inference || rate === 0) {
        return x;
    }

    return new Tensor(native.dropoutOp(x._id, rate, true));
}

export function crossEntropyLoss(logits: Tensor, targets: number[][] | Tensor): Tensor {
    const shape = logits.shape;
    const vocabSize = shape[shape.length - 1]!;
    const batchTimesTokens = logits.size / vocabSize;
    const flatLogits = logits.view(batchTimesTokens, vocabSize);
    const flatTargets = flattenClassTargets(targets, vocabSize);

    return new Tensor(native.crossEntropyLoss(flatLogits._id, flatTargets));
}

export function layerNorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: number = 1e-5): Tensor {
    return new Tensor(native.layernormOp(x._id, gamma._id, beta._id, eps));
}

export function mseLoss(pred: Tensor, target: Tensor): Tensor {
    const diff = pred.sub(target);
    const sq = diff.mul(diff);
    return new Tensor(native.meanOp(new Tensor(native.meanOp(sq._id, -1))._id, -1));
}

export function logsoftmax(x: Tensor, dim: number = -1): Tensor {
    return softmax(x, dim).log();
}

export function randRange(shape: number[], min: number, max: number): Tensor {
    const randomTensor = Tensor.rand(shape);
    const data = randomTensor.toFloat32();
    const range = max - min;
    const scaled = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
        scaled[i] = data[i]! * range + min;
    }

    const tensor = Tensor.fromFloat32(scaled, shape);
    tensor.setRequiresGrad(true);
    return tensor;
}

function parseKernelSize(
    kernelSizeOrHeight: [number, number] | number,
    maybeWidth?: number,
): [number, number] {
    if (Array.isArray(kernelSizeOrHeight)) {
        return kernelSizeOrHeight;
    }

    return [kernelSizeOrHeight, maybeWidth ?? kernelSizeOrHeight];
}

export function tile(_x: Tensor, _reps: number[]): Tensor {
    throw new Error("tile not implemented in native backend");
}

export function avgpool2d(
    _x: Tensor,
    kernelSizeOrHeight: [number, number] | number,
    maybeWidth?: number,
): Tensor {
    const [_kernelHeight, _kernelWidth] = parseKernelSize(kernelSizeOrHeight, maybeWidth);
    throw new Error("avgpool2d not implemented in native backend");
}

export function maxpool2d(
    _x: Tensor,
    kernelSizeOrHeight: [number, number] | number,
    maybeWidth?: number,
): Tensor {
    const [_kernelHeight, _kernelWidth] = parseKernelSize(kernelSizeOrHeight, maybeWidth);
    throw new Error("maxpool2d not implemented in native backend");
}

