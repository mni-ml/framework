import { Module, Parameter } from '../../toy/module.js';
import { Tensor } from '../../toy/tensor.js';
import { TensorHistory } from '../../toy/tensor_functions.js';
import { mseLoss } from '../../toy/nn.js';

export class VectorQuantize extends Module {
    codebook!: Parameter<Tensor>;
    numCodes: number;
    dim: number;

    constructor(dim: number, numCodes: number = 64) {
        super();
        this.dim = dim;
        this.numCodes = numCodes;
        const cb = Tensor.randn([numCodes, dim]).mul(1 / Math.sqrt(dim));
        cb.history = new TensorHistory();
        this.codebook = new Parameter(cb);
    }

    forward(x: Tensor): { quantized: Tensor; indices: number[]; commitmentLoss: Tensor } {
        // x: [N, dim]
        const N = x.shape[0]!;
        const cb = this.codebook.value;

        // Compute pairwise distances: ||x - c||^2 = ||x||^2 - 2*x*c^T + ||c||^2
        const xSq = x.mul(x).sum(-1);               // [N, 1]
        const cbSq = cb.mul(cb).sum(-1);             // [numCodes, 1]
        const xCb = x.matmul(cb.transpose());        // [N, numCodes]

        // distances[i][j] = xSq[i] - 2*xCb[i][j] + cbSq[j]
        // Find argmin per row (manual since we don't have argmin)
        const indices: number[] = [];
        for (let i = 0; i < N; i++) {
            let minDist = Infinity;
            let minIdx = 0;
            for (let j = 0; j < this.numCodes; j++) {
                const dist = xSq.get([i, 0]) - 2 * xCb.get([i, j]) + cbSq.get([j, 0]);
                if (dist < minDist) {
                    minDist = dist;
                    minIdx = j;
                }
            }
            indices.push(minIdx);
        }

        // Gather quantized vectors from codebook
        const quantizedData = new Float64Array(N * this.dim);
        for (let i = 0; i < N; i++) {
            for (let d = 0; d < this.dim; d++) {
                quantizedData[i * this.dim + d] = cb.get([indices[i]!, d]);
            }
        }
        const quantizedRaw = Tensor.tensor(Array.from(quantizedData), [N, this.dim]);

        // Straight-through estimator: forward uses quantized, backward flows through x
        const quantized = x.add(quantizedRaw.sub(x).detach());

        // Commitment loss: encourages encoder output to stay close to codebook
        const commitmentLoss = mseLoss(x.detach(), quantizedRaw);

        return { quantized, indices, commitmentLoss };
    }
}
