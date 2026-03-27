import "./operators.js"
import { Context } from "./autodiff.js"
import { fastTensorReduce } from "./fast_ops.js"
import { Tensor } from "./tensor.js"
import { TensorData } from "./tensor_data.js"
import "./tensor_functions.js"

// # List of functions in this file:
// # - avgpool2d: Tiled average pooling 2D
// # - argmax: Compute the argmax as a 1-hot tensor
// # - Max: New Function for max operator
// # - max: Apply max reduction
// # - softmax: Compute the softmax as a tensor
// # - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
// # - maxpool2d: Tiled max pooling 2D
// # - dropout: Dropout positions based on random noise, include an argument to turn off

export function tile(input: Tensor, kernel: [number, number]): [Tensor, number, number] {
    const [batch, channel, height, width] = input.shape;
    const [kh, kw] = kernel;

    if (!batch) throw new Error("input width undefined");
    if (!channel) throw new Error("input width undefined");
    if (!height) throw new Error("input height undefined");
    if (!width) throw new Error("input width undefined");

    // ensure the dimensions for pooling are divisible by the pooling constant
    if (height % kh !== 0) {
        throw new Error("input height must be divisible by kernel height");
    }
    if (width % kw !== 0) {
        throw new Error("input width must be divisible by kernel width");
    }
    
    const newHeight = height / kh;
    const newWidth = width / kh;

    // add an extra dimension by applying view.
    const tiled = input.view(batch, channel, newHeight, kh, newWidth, kw);

    // use other functions such as avgpool2d to reduce over the tiled tensor.

    return [tiled, newHeight, newWidth];
}


export function avgpool2d(input: Tensor, kernel: [number, number]): Tensor {
    const [tiled, newHeight, newWidth]: [Tensor, number, number] = tile(input, kernel);
    const [kh, kw] = kernel;

    // [batch, channel, new height, kh, new width, kw] -> [batch, channel, new height, new width, kh, kw]
    // note the swapping of index 3 & 4
    const perm = tiled.permute(0, 1, 2, 4, 3, 5);

    const [batch, channel] = perm.shape as [number, number, number, number, number, number];

    const inputData = new TensorData(perm.data.storage, perm.shape);
    const outputData = TensorData.zeros(perm.shape);

    const acc = (acc: number, x: number) => (acc + x);
    const reduceFn = fastTensorReduce((acc: number, x: number) => acc + x);

    // for 2d convolution pooling, we reduce over both cols and rows

    // sum over kw. note the replacement of kw with 1. also note that 5 is dimension index
    const sumKw = TensorData.zeros([batch, channel, newHeight, newWidth, kh, 1])
    reduceFn(sumKw.storage, sumKw.shape, sumKw.strides, inputData.storage, inputData.shape, inputData.strides, 5);

    // sum over kh
    const sumKh = TensorData.zeros([batch, channel, newHeight, newWidth, 1, 1])
    reduceFn(sumKh.storage, sumKh.shape, sumKh.strides, inputData.storage, inputData.shape, inputData.strides, 4);

    // return new Tensor(sumKh.data.storage);
}