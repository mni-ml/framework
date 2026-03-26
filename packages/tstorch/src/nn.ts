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
