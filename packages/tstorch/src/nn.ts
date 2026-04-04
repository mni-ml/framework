import { Tensor } from "./tensor.js"

/**
 * Reshape an image tensor for 2D pooling.
 *
 * input:  batch x channel x height x width
 * kernel: [kh, kw]
 * returns: tensor of shape [batch, channel, newHeight, newWidth, kh*kw],
 *          plus the newHeight and newWidth values.
 */
export function tile(input: Tensor, kernel: [number, number]): [Tensor, number, number] {
    const [batch, channel, height, width] = input.shape;
    const [kh, kw] = kernel;

    if (!batch) throw new Error("input batch undefined");
    if (!channel) throw new Error("input channel undefined");
    if (!height) throw new Error("input height undefined");
    if (!width) throw new Error("input width undefined");

    if (height % kh !== 0) {
        throw new Error("input height must be divisible by kernel height");
    }
    if (width % kw !== 0) {
        throw new Error("input width must be divisible by kernel width");
    }

    const newHeight = height / kh;
    const newWidth = width / kw;

    const tiled = input.contiguous()
        .view(batch, channel, newHeight, kh, newWidth, kw)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(batch, channel, newHeight, newWidth, kh * kw);

    return [tiled, newHeight, newWidth];
}

export function avgpool2d(input: Tensor, kernel: [number, number]): Tensor {
    const [batch, channel] = input.shape;
    const [tiled, newHeight, newWidth] = tile(input, kernel);
    return tiled.mean(4).view(batch!, channel!, newHeight, newWidth);
}

export function maxpool2d(input: Tensor, kernel: [number, number]): Tensor {
    const [batch, channel] = input.shape;
    const [tiled, newHeight, newWidth] = tile(input, kernel);
    return tiled.max(4).view(batch!, channel!, newHeight, newWidth);
}

export function softmax(input: Tensor, dim: number): Tensor {
    const m = input.max(dim);
    const e = input.sub(m).exp();
    return e.mul(e.sum(dim).inv());
}

export function logsoftmax(input: Tensor, dim: number): Tensor {
    const m = input.max(dim);
    const shifted = input.sub(m);
    const logSumExp = shifted.exp().sum(dim).log();
    return shifted.sub(logSumExp);
}

export function dropout(input: Tensor, rate: number = 0.5, ignore: boolean = false): Tensor {
    if (ignore || rate === 0.0) return input;
    if (rate >= 1.0) return Tensor.zeros(input.shape);
    const mask = Tensor.rand(input.shape).gt(rate);
    return input.mul(mask).mul(1 / (1 - rate));
}